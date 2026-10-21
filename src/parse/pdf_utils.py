"""PyMuPDF helpers shared by the PDF parsers.

The main problem these helpers solve is PDFs that have been "printed" by a web
browser (e.g. the NCBI Bookshelf copy of the WHO hypertension guideline). Such
files embed every page's text as Type3 fonts whose ``ToUnicode`` maps ligatures
and accents to Private Use Area (PUA) code points. The mapping is *per font*,
so the same PUA code point can mean "fi" in one font and "fl" in the next.

We therefore identify each PUA glyph by the MD5 hash of its Type3 ``CharProcs``
drawing stream, which is identical wherever the same glyph is drawn, and let the
caller supply a ``{glyph_hash: replacement}`` table. Unknown PUA glyphs raise
an error rather than being silently dropped.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pymupdf

PUA_RE = re.compile("[-]")
_FONT_XREF_RE = re.compile(r"\((\d+) 0 R\)")


class UnknownGlyphError(ValueError):
    """Raised when a PUA glyph is met that has no entry in the glyph table."""


@dataclass(frozen=True)
class Span:
    """A run of text with uniform styling, after PUA glyph resolution."""

    text: str
    bbox: tuple[float, float, float, float]
    size: float
    color: int
    font: str
    flags: int
    direction: tuple[float, float]
    page: int  # 1-based PDF page number

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def horizontal(self) -> bool:
        return abs(self.direction[0] - 1.0) < 1e-3 and abs(self.direction[1]) < 1e-3


@dataclass
class Line:
    """A visual line: spans sharing a baseline, ordered left to right."""

    spans: list[Span]

    @property
    def text(self) -> str:
        """Concatenate spans, inserting a space where there is a visible gap."""
        parts: list[str] = []
        prev: Span | None = None
        for span in self.spans:
            if prev is not None:
                gap = span.x0 - prev.x1
                if gap > 1.0 and not parts[-1].endswith(" ") and not span.text.startswith(" "):
                    parts.append(" ")
            parts.append(span.text)
            prev = span
        return "".join(parts)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (
            min(s.x0 for s in self.spans),
            min(s.y0 for s in self.spans),
            max(s.x1 for s in self.spans),
            max(s.y1 for s in self.spans),
        )

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]

    @property
    def page(self) -> int:
        return self.spans[0].page

    @property
    def main_span(self) -> Span:
        """The span carrying most of the line's visible text (for styling decisions)."""
        return max(self.spans, key=lambda s: len(s.text.strip()))

    @property
    def size(self) -> float:
        return self.main_span.size

    @property
    def color(self) -> int:
        return self.main_span.color


def _parse_tounicode(cmap: str) -> dict[int, str]:
    """Parse ``bfchar`` and ``bfrange`` entries of a ToUnicode CMap into ``{code: text}``."""
    mapping: dict[int, str] = {}

    def decode(hexstr: str) -> str:
        return bytes.fromhex(hexstr).decode("utf-16-be", errors="replace")

    for block in re.findall(r"beginbfchar(.*?)endbfchar", cmap, re.S):
        for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            mapping[int(src, 16)] = decode(dst)
    for block in re.findall(r"beginbfrange(.*?)endbfrange", cmap, re.S):
        for lo, hi, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
            start = int(dst, 16)
            for offset, code in enumerate(range(int(lo, 16), int(hi, 16) + 1)):
                mapping[code] = chr(start + offset)
    return mapping


def type3_pua_glyph_hashes(doc: pymupdf.Document) -> dict[tuple[int, str], str]:
    """Map ``(font xref, PUA character)`` to the MD5 of the glyph's drawing stream.

    Only Type3 fonts with a ``ToUnicode`` CMap are examined; only PUA targets are
    returned. The hash is truncated to 8 hex characters, which is ample for the
    handful of distinct glyphs in a document.
    """
    result: dict[tuple[int, str], str] = {}
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "Subtype") != ("name", "/Type3"):
                continue
        except Exception:  # pragma: no cover - malformed objects
            continue
        kind, value = doc.xref_get_key(xref, "ToUnicode")
        if kind != "xref":
            continue
        cmap = doc.xref_stream(int(value.split()[0])).decode("latin-1")
        to_unicode = _parse_tounicode(cmap)
        obj = doc.xref_object(xref)

        diffs = re.search(r"/Differences\s*\[(.*?)\]", obj, re.S)
        code_to_name: dict[int, str] = {}
        if diffs:
            code = 0
            for token in diffs.group(1).split():
                if token.startswith("/"):
                    code_to_name[code] = token[1:]
                    code += 1
                else:
                    code = int(token)
        procs_part = obj.split("/CharProcs", 1)
        name_to_xref = dict(re.findall(r"/([^\s/<>\[\]]+)\s+(\d+)\s+0\s+R", procs_part[1])) if len(procs_part) == 2 else {}

        for code, text in to_unicode.items():
            if not text or not PUA_RE.fullmatch(text[0]):
                continue
            name = code_to_name.get(code)
            proc_xref = name_to_xref.get(name) if name else None
            if proc_xref is None:
                continue
            digest = hashlib.md5(doc.xref_stream(int(proc_xref))).hexdigest()[:8]
            result[(xref, text[0])] = digest
    return result


def page_label_map(doc: pymupdf.Document) -> dict[str, int]:
    """Map each page label defined in the PDF (e.g. printed page "290") to its 1-based PDF page.

    Uses the document's own ``/PageLabels``, so no page offset is assumed. Pages
    without a label are omitted; if a label occurs twice, the first page wins.
    """
    labels: dict[str, int] = {}
    for page in doc:
        label = page.get_label()
        if label and label not in labels:
            labels[label] = page.number + 1
    return labels


def _font_xref(font_name: str) -> int | None:
    match = _FONT_XREF_RE.search(font_name)
    return int(match.group(1)) if match else None


def extract_spans(
    page: pymupdf.Page,
    pua_hashes: dict[tuple[int, str], str],
    glyph_table: dict[str, str],
) -> list[Span]:
    """Extract all text spans on ``page`` with PUA glyphs replaced via ``glyph_table``.

    Args:
        page: PyMuPDF page.
        pua_hashes: output of :func:`type3_pua_glyph_hashes` for the document.
        glyph_table: ``{glyph_hash: replacement}``; replacements may be combining
            marks, which :func:`src.parse.text_utils.normalise_text` composes with
            NFC later.

    Raises:
        UnknownGlyphError: if a PUA glyph cannot be resolved.
    """
    spans: list[Span] = []
    raw = page.get_text("rawdict", flags=pymupdf.TEXTFLAGS_RAWDICT & ~pymupdf.TEXT_PRESERVE_IMAGES)
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            direction = tuple(line["dir"])
            for s in line["spans"]:
                text = "".join(c["c"] for c in s["chars"])
                if not text:
                    continue
                if PUA_RE.search(text):
                    fx = _font_xref(s["font"])
                    out = []
                    for ch in text:
                        if not PUA_RE.fullmatch(ch):
                            out.append(ch)
                            continue
                        digest = pua_hashes.get((fx, ch)) if fx is not None else None
                        if digest is None or digest not in glyph_table:
                            raise UnknownGlyphError(
                                f"Unresolved PUA glyph U+{ord(ch):04X} (font {s['font']}, hash {digest}) "
                                f"on page {page.number + 1}: {text!r}"
                            )
                        out.append(glyph_table[digest])
                    text = "".join(out)
                spans.append(
                    Span(
                        text=text,
                        bbox=tuple(s["bbox"]),
                        size=round(s["size"], 1),
                        color=s["color"],
                        font=s["font"],
                        flags=s["flags"],
                        direction=direction,
                        page=page.number + 1,
                    )
                )
    return spans


def group_lines(spans: list[Span], y_tolerance: float = 2.5) -> list[Line]:
    """Group horizontal spans into visual lines by baseline, ordered top to bottom.

    Two spans belong to the same line when their bottoms differ by at most
    ``y_tolerance`` points. Non-horizontal spans (e.g. rotated side tabs) are
    dropped.
    """
    horizontal = sorted((s for s in spans if s.horizontal), key=lambda s: (round(s.y1, 1), s.x0))
    lines: list[Line] = []
    for span in horizontal:
        if lines and abs(lines[-1].spans[-1].y1 - span.y1) <= y_tolerance:
            lines[-1].spans.append(span)
        else:
            lines.append(Line([span]))
    for line in lines:
        line.spans.sort(key=lambda s: s.x0)
    return lines
