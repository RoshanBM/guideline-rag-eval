"""Parse the WHO hypertension guideline (2021) into ``data/chunks/hypertension.jsonl``.

Output: one chunk per recommendation *statement*, prose chunks for the other
substantive sections (split by heading, ~150–400 words), table chunks for
Table 1 and Fig. A3.1, and "pico" chunks for the Annex 4 PICO questions
(``index_default: false``, so retrieval skips them unless asked).

Run with::

    python -m src.parse.who_hypertension

Design notes (see ``notes/source_inspection.md`` for the evidence):

* The file is the NCBI Bookshelf copy printed from a browser. Text is in
  Type3 fonts whose ligatures/accents are Private Use Area code points with a
  per-font meaning; they are resolved by glyph-drawing hash (``GLYPH_TABLE``).
* Running headers, page numbers and rotated side tabs are removed by colour,
  position and text direction. Headings are detected by size and colour
  because the fonts carry no bold/italic flags.
* ``strength_as_printed`` and ``certainty`` are read from the grading line
  beneath each statement in section 3, and cross-checked against the Executive
  summary and the GRADE wording convention ("WHO recommends" = strong, "WHO
  suggests" = conditional). ``strength`` equals the printed value unless a
  conflict has been reviewed and resolved in ``GRADING_RESOLUTIONS``, in which
  case the resolution and its evidence are stored in ``source_discrepancy``.
  Unreviewed conflicts give ``strength = None``.
* Each chunk carries both the 1-based PDF page and the page number printed on
  that page (read from the page-number tab, not computed from an offset).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from src.parse.pdf_utils import Line, Span, extract_spans, group_lines, type3_pua_glyph_hashes
from src.parse.sources import CHUNKS_DIR, LICENCE, get_source
from src.parse.text_utils import build_vocabulary, join_lines, normalise_text

# --------------------------------------------------------------------------- #
# Document constants
# --------------------------------------------------------------------------- #
DOC_ID = "who-htn-2021"
DOC_TITLE = "WHO Guideline for the pharmacological treatment of hypertension in adults (2021)"
COLLECTION = "cardiometabolic"
# The PDF prints no IRIS handle or URL (only "CIP data are available at
# http://apps.who.int/iris"), so the handle supplied with the project brief is used.
SOURCE_URL = "https://iris.who.int/handle/10665/344424"
OUTPUT_PATH = CHUNKS_DIR / "hypertension.jsonl"

MIN_WORDS = 150
MAX_WORDS = 400

# Type3 CharProcs MD5 (first 8 hex chars) -> replacement text. Determined by
# inspection: each hash was checked against the words it occurs in, and the
# dotless-i glyph visually (see notes/source_inspection.md, section 3.3).
GLYPH_TABLE: dict[str, str] = {
    "6a23a277": "fi",
    "169a766e": "fi",
    "8ffb1680": "fi",
    "e71e9475": "fl",
    "f0f57f4b": "fl",
    "a846d610": "́",  # spacing acute after its base letter -> combining acute
    "2dce5ce9": "́",
    "3153d706": "̀",  # grave
    "29b7749d": "̀",
    "a0fbcf41": "̈",  # diaeresis
    "44d02e14": "i",  # dotless i used as the base of "í" (Martí)
}

# Colours (sRGB ints) and sizes measured from the PDF.
RED = 0xB6294C
BLACK = 0x221F21
WHITE = 0xFDFEFF
BLUE = 0x0093D5
HEADER_GREY = 0x6B6E71
CAPTION_GREY = 0x56585B

# Chapters (level-1 headings) whose content is chunked. Everything else is front
# matter, references or the list of contributors (Annex 1) and is skipped. The
# Executive summary is read only to cross-check recommendation gradings.
INCLUDED_CHAPTER_PREFIXES = ("1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "Annex 2", "Annex 3", "Annex 4")
EXEC_SUMMARY_TITLE = "Executive summary"
# Analytic-framework diagrams: their labels extract as loose fragments and their
# content is given in prose (sections 2.2, 3.2) and in Annex 4, so they are skipped.
SKIPPED_FIGURES = ("Fig. 1", "Fig. 2")

@dataclass(frozen=True)
class Resolution:
    """A reviewed decision on a grading that the source states inconsistently."""

    printed: str  # value printed beneath the statement; the resolution applies only if this still matches
    resolved: str
    note: str  # stored verbatim in the chunk's ``source_discrepancy``


# Grading conflicts that have been reviewed and resolved, keyed by (recommendation, rec_part).
# Any conflict NOT listed here is stored as strength = None and reported as unresolved.
GRADING_RESOLUTIONS: dict[tuple[int, int], Resolution] = {
    (1, 3): Resolution(
        printed="strong",
        resolved="conditional",
        note=(
            "Recommendation box (printed p. 7) states 'Strong recommendation'; Executive summary "
            "(printed p. viii) states 'Conditional recommendation'. Wording 'WHO suggests' indicates "
            "conditional. Al-Makki et al., Hypertension 2022;79:293–301 "
            "(doi:10.1161/HYPERTENSIONAHA.121.18192), Figure 3, states 'Conditional recommendation, "
            "moderate- to high-quality evidence'. NCBI Bookshelf HTML version not yet checked "
            "(automated access blocked). Resolved as conditional."
        ),
    ),
}

GRADING_RE = re.compile(r"^(Strong|Conditional) recommendation,\s*(.+?)\s*evidence\.?$")
CERTAINTY_RE = re.compile(r"^(?:(high|moderate|low|very low)-?\s+to\s+)?(high|moderate|low|very low)-certainty$")
REC_TITLE_RE = re.compile(r"^(\d+)\.\s+RECOMMENDATIONS?\s+ON\b")
CAPTION_RE = re.compile(r"^(Fig\. ?A?\d+(?:\.\d+)?|Table \d+)")
BULLET_RE = re.compile(r"^\[\s*")
NUMBERED_ITEM_RE = re.compile(r"^\d+\.\s")
ALGORITHM_RE = re.compile(r"^Algorithm \d+:")


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    """An atomic unit of chunk text (paragraph, list item or whole figure)."""

    text: str
    page_start: int
    page_end: int
    starts_paragraph: bool = True

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Leaf:
    """Content under one heading path, before merging/splitting into chunks."""

    path: list[str]
    blocks: list[Block] = field(default_factory=list)
    standalone: bool = False  # never merged with neighbours (implementation remarks)
    seq: int = 0  # number of recommendation statements seen before this leaf
    slug: str | None = None  # chunk-id slug override for leaves merged across sections
    parent_rec: int | None = None  # recommendation number, for implementation remarks

    @property
    def words(self) -> int:
        return sum(b.words for b in self.blocks)


@dataclass
class Statement:
    rec_number: int
    rec_part: int  # 1-based position of the statement within its recommendation box
    title: str
    text: str
    grading: str
    page_start: int
    page_end: int
    section_path: list[str]


@dataclass
class Chunk:
    chunk_id: str
    section_path: list[str]
    chunk_type: str  # "recommendation" | "prose" | "table" | "pico"
    text: str
    page_start: int  # 1-based PDF page
    page_end: int
    rec_id: str | None = None
    rec_part: int | None = None
    parent_rec_id: str | None = None
    strength: str | None = None  # resolved strength (equals the printed one unless a discrepancy was resolved)
    strength_as_printed: str | None = None  # grading printed beneath the statement in section 3
    certainty: str | None = None
    source_discrepancy: str | None = None  # how a conflict between parts of the source was resolved
    printed_page_start: int | None = None  # page number printed on the page itself
    printed_page_end: int | None = None

    @property
    def index_default(self) -> bool:
        """PICO question tables are excluded from retrieval unless asked for."""
        return self.chunk_type != "pico"

    def to_record(self) -> dict[str, object]:
        """Return the record in the exact key order of the project schema (see README)."""
        return {
            "chunk_id": self.chunk_id,
            "collection": COLLECTION,
            "doc": DOC_TITLE,
            "doc_id": DOC_ID,
            "section_path": self.section_path,
            "rec_id": self.rec_id,
            "rec_part": self.rec_part,
            "parent_rec_id": self.parent_rec_id,
            "chunk_type": self.chunk_type,
            "index_default": self.index_default,
            "strength": self.strength,
            "strength_as_printed": self.strength_as_printed,
            "certainty": self.certainty,
            "source_discrepancy": self.source_discrepancy,
            "text": self.text,
            "pdf_page_start": self.page_start,
            "pdf_page_end": self.page_end,
            "printed_page_start": self.printed_page_start,
            "printed_page_end": self.printed_page_end,
            "source_url": SOURCE_URL,
            "licence": LICENCE,
        }


@dataclass
class ParseReport:
    """Facts collected during parsing, printed in the summary."""

    conflicts: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Page reading
# --------------------------------------------------------------------------- #
def is_furniture(span: Span) -> bool:
    """Running header/footer, page-number tab or rotated side tab."""
    if not span.horizontal:
        return True
    if span.color == HEADER_GREY:
        return True
    return span.y0 > 800 and span.color == WHITE


ROMAN_RE = re.compile(r"^[ivxlc]+$")


def printed_page_numbers(doc: pymupdf.Document) -> tuple[dict[int, int | None], set[int]]:
    """Read the page number printed on each page (white text in the tab at the page foot).

    Returns ``(numbers, roman_pages)``: ``numbers`` maps every 1-based PDF page to
    its printed arabic page number, or ``None`` if the page prints no number
    (cover, blank pages, ISBN page). Front-matter pages numbered in roman
    numerals map to ``None`` and are listed in ``roman_pages``; none of them is
    chunked, and :func:`parse` checks that.
    """
    numbers: dict[int, int | None] = {}
    roman: set[int] = set()
    for page in doc:
        found: list[str] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    text = s["text"].strip()
                    if s["bbox"][1] > 800 and s["color"] == WHITE and (text.isdigit() or ROMAN_RE.match(text)):
                        found.append(text)
        if len(found) > 1:
            raise ValueError(f"PDF page {page.number + 1}: several page numbers found: {found}")
        pno = page.number + 1
        if found and found[0].isdigit():
            numbers[pno] = int(found[0])
        else:
            numbers[pno] = None
            if found:
                roman.add(pno)
    return numbers, roman


def body_margin(page_number: int) -> float:
    """Left text margin: odd (recto) pages 79.4 pt, even (verso) pages 70.9 pt."""
    return 79.4 if page_number % 2 else 70.9


def is_heading(line: Line) -> int:
    """Return heading level 1–3 for chapter/section/sub-heading lines, else 0."""
    span = line.main_span
    if span.color != RED:
        return 0
    if span.size >= 20:
        return 1
    if span.size == 12.0:
        return 2
    if span.size == 10.0 and not BULLET_RE.match(line.text.strip()):
        return 3
    return 0


def is_body_resume(span: Span) -> bool:
    """A span that marks the end of a figure/table region (body text or heading)."""
    if span.color == BLACK and span.size == 10.0:
        return True
    return span.color == RED and span.size >= 12.0


@dataclass
class Figure:
    caption: str
    page: int
    y0: float
    spans: list[Span]


def split_figures(page_spans: list[Span]) -> tuple[list[Span], list[Figure]]:
    """Separate figure/table regions from running text on one page.

    A region starts at a grey caption ("Fig. N ..." / "Table N ...") and runs
    until body text or a heading resumes, or to the end of the page.
    """
    spans = sorted(page_spans, key=lambda s: (round(s.y0, 1), s.x0))
    text_spans: list[Span] = []
    figures: list[Figure] = []
    current: Figure | None = None
    for span in spans:
        is_caption = span.color == CAPTION_GREY and span.size == 10.0
        if is_caption and (CAPTION_RE.match(span.text.strip()) or current is not None and not current.spans):
            if current is not None and not current.spans:
                current.caption += " " + span.text.strip()  # wrapped caption line
            else:
                current = Figure(span.text.strip(), span.page, span.y0, [])
                figures.append(current)
            continue
        if current is not None:
            if is_body_resume(span) and span.y0 > current.y0:
                current = None
            else:
                current.spans.append(span)
                continue
        text_spans.append(span)
    return text_spans, figures


# --------------------------------------------------------------------------- #
# Line -> paragraph assembly
# --------------------------------------------------------------------------- #
def strip_bullet(text: str) -> tuple[bool, str]:
    """Detect the bullet glyph (extracted as '[') at a line start."""
    stripped = text.strip()
    if BULLET_RE.match(stripped) and stripped != "[":
        return True, "• " + BULLET_RE.sub("", stripped)
    return False, stripped


def lines_to_blocks(lines: list[Line], vocab: set[str], para_gap: float = 7.5) -> list[Block]:
    """Group consecutive lines into paragraph / list-item blocks.

    A new block starts at a bullet, a numbered list item, or a vertical gap larger
    than ``para_gap`` points. Across a page break a paragraph continues only if the
    previous line does not end a sentence and the next starts in lower case.
    """
    blocks: list[Block] = []
    buf: list[str] = []
    pages: list[int] = []
    prev: Line | None = None

    def flush() -> None:
        if buf:
            text = normalise_text(join_lines(buf, vocab))
            if text:
                blocks.append(Block(text, pages[0], pages[-1]))
        buf.clear()
        pages.clear()

    for line in lines:
        bullet, text = strip_bullet(line.text)
        if not text or text == "[":
            continue
        new_para = bullet or bool(NUMBERED_ITEM_RE.match(text))
        if prev is not None and not new_para:
            if line.page != prev.page:
                last = buf[-1] if buf else ""
                new_para = bool(re.search(r"[.:;?!)]$", last)) or not text[:1].islower()
            elif line.y0 - prev.y1 > para_gap:
                new_para = True
        if new_para:
            flush()
        buf.append(text)
        pages.append(line.page)
        prev = line
    flush()
    return blocks


# --------------------------------------------------------------------------- #
# Figure and table formatting
# --------------------------------------------------------------------------- #
def _is_glyph(span: Span) -> bool:
    """A lone symbol (≥, bullet) set in a symbol font with its own size/colour."""
    t = span.text.strip()
    return len(t) == 1 and not t.isalnum()


def cluster_boxes(spans: list[Span], max_gap: float = 4.0) -> list[list[Span]]:
    """Cluster figure spans into boxes.

    Text spans join a box when they share colour and size, touch vertically and
    overlap horizontally. Lone symbol glyphs (≥, bullets) come from a symbol font
    with different styling, so they join whatever text sits beside them on the
    same baseline instead.
    """
    spans = sorted(spans, key=lambda s: (s.y0, s.x0))
    parent = list(range(len(spans)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(spans):
        for j in range(i + 1, len(spans)):
            b = spans[j]
            if b.y0 - a.y1 > max_gap:
                break
            if _is_glyph(a) or _is_glyph(b):
                overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0) > 0
                gap_x = max(a.x0, b.x0) - min(a.x1, b.x1)
                linked = overlap_y and gap_x <= 15
            else:
                same_style = a.color == b.color and a.size == b.size
                linked = same_style and min(a.x1, b.x1) - max(a.x0, b.x0) > 0
            if linked:
                parent[find(j)] = find(i)
    groups: dict[int, list[Span]] = {}
    for i, s in enumerate(spans):
        groups.setdefault(find(i), []).append(s)
    return list(groups.values())


def box_text(spans: list[Span], vocab: set[str]) -> list[str]:
    """Text of one box as one or more paragraphs (list items split out)."""
    lines = group_lines([s for s in spans if s.text.strip()])
    return [b.text for b in lines_to_blocks(lines, vocab, para_gap=6.0)]


def flowchart_block(fig: Figure, vocab: set[str]) -> Block:
    """Linearise Figs 3–6.

    Main boxes (white text) are read top to bottom. Each blue "Recheck BP ..."
    box is placed directly after the main box it sits beside (largest vertical
    overlap). Notes, warnings and footnotes follow in vertical order.
    """
    boxes = cluster_boxes(fig.spans)
    def colour(box: list[Span]) -> int:
        return max(box, key=lambda s: len(s.text.strip())).color

    main = sorted((b for b in boxes if colour(b) == WHITE), key=lambda b: min(s.y0 for s in b))
    side = [b for b in boxes if colour(b) == BLUE]
    rest = sorted((b for b in boxes if colour(b) not in (WHITE, BLUE)), key=lambda b: min(s.y0 for s in b))

    def yr(box: list[Span]) -> tuple[float, float]:
        return min(s.y0 for s in box), max(s.y1 for s in box)

    attached: dict[int, list[list[Span]]] = {i: [] for i in range(len(main))}
    for sb in side:
        s0, s1 = yr(sb)
        best = max(range(len(main)), key=lambda i: min(s1, yr(main[i])[1]) - max(s0, yr(main[i])[0]))
        attached[best].append(sb)

    paragraphs = [fig.caption]
    for i, box in enumerate(main):
        paragraphs += box_text(box, vocab)
        for sb in attached[i]:
            paragraphs += box_text(sb, vocab)
    for box in rest:
        paragraphs += box_text(box, vocab)
    text = normalise_text("\n".join(paragraphs))
    return Block(text, fig.page, fig.page)


def certainty_table(fig: Figure, vocab: set[str]) -> str:
    """Table 1: two columns (certainty level | definition), rows top-aligned."""
    lines = group_lines(fig.spans)
    header_spans = [s for s in fig.spans if s.color == WHITE]
    header = " | ".join(s.text.strip() for s in sorted(header_spans, key=lambda s: s.x0))
    split_x = min(s.x0 for s in header_spans if s.x0 > min(h.x0 for h in header_spans))
    rows: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.main_span.color == WHITE:
            continue
        left = " ".join(s.text.strip() for s in line.spans if s.x0 < split_x - 5).strip()
        right = " ".join(s.text.strip() for s in line.spans if s.x0 >= split_x - 5).strip()
        if left:
            rows.append((left, []))
        if right:
            rows[-1][1].append(right)
    out = [header] + [f"{label} | {join_lines(defn, vocab)}" for label, defn in rows]
    return normalise_text("\n".join(out))


def ratings_table(fig: Figure) -> str:
    """Fig. A3.1: pair each outcome label (bar text) with its rating (bubble) by baseline."""
    spans = [s for s in fig.spans if s.text.strip()]
    title = [s for s in spans if s.color == BLACK and s.y0 < min(x.y0 for x in spans if x.color == WHITE)]
    axis = [s for s in spans if s.color == BLACK and s not in title]
    values = [s for s in spans if s.color == WHITE and re.fullmatch(r"\d+\.\d", s.text.strip())]
    labels = [s for s in spans if s.color == WHITE and s not in values]
    rows = []
    for lab in sorted(labels, key=lambda s: s.y0):
        val = min(values, key=lambda v: abs(v.y1 - lab.y1))
        if abs(val.y1 - lab.y1) > 3:
            raise ValueError(f"No rating aligned with outcome {lab.text!r}")
        rows.append(f"{lab.text.strip()} | {val.text.strip()}")
    if len(rows) != len(values):
        raise ValueError("Fig. A3.1: label/value count mismatch")
    # Chart title, one "outcome | rating" row per bar, then the axis labels.
    out = [" ".join(s.text.strip() for s in title), *rows]
    out.append(" | ".join(s.text.strip() for s in sorted(axis, key=lambda s: s.x0)))
    return normalise_text("\n".join(out))


def pico_tables(lines: list[Line], vocab: set[str]) -> list[tuple[int, str, int, int]]:
    """Annex 4: one table per PICO question -> ``(number, text, page_start, page_end)``.

    Each question is an 11 pt red heading, followed by rows whose label (P, I, C,
    O, s) is in a narrow left column. Labels are sometimes vertically centred,
    sometimes top-aligned, so rows are delimited by the larger vertical gap
    between rows (>= 16 pt) and then matched to labels.
    """
    questions: list[dict] = []
    for line in lines:
        span = line.main_span
        if span.color == RED and span.size == 11.0:
            text = line.text.strip()
            m = re.match(r"^(\d+)\s+(.*)$", text)
            if m:
                questions.append({"n": int(m.group(1)), "title": [m.group(2)], "labels": [], "content": [], "notes": [], "pages": [line.page]})
            else:
                questions[-1]["title"].append(text)
            continue
        if not questions:
            continue
        q = questions[-1]
        q["pages"].append(line.page)
        label_spans = [s for s in line.spans if s.size == 10.0 and s.text.strip() in {"P", "I", "C", "O", "s"}]
        for s in label_spans:
            q["labels"].append(s)
        content = [s for s in line.spans if s not in label_spans and s.text.strip()]
        if not content:
            continue
        if content[0].color == CAPTION_GREY:
            q["notes"].append(" ".join(s.text.strip() for s in content))
        else:
            q["content"].append(Line(content))

    out = []
    for q in questions:
        groups: list[list[Line]] = []
        for line in q["content"]:
            if groups and (line.page == groups[-1][-1].page and line.y0 - groups[-1][-1].y1 < 7.0):
                groups[-1].append(line)
            else:
                groups.append([line])
        labels = sorted(q["labels"], key=lambda s: (s.page, s.y0))
        if len(groups) != len(labels):
            raise ValueError(f"PICO {q['n']}: {len(groups)} row groups but {len(labels)} labels")
        rows = []
        for lab, grp in zip(labels, groups):
            if not (grp[0].y0 - 3 <= (lab.y0 + lab.y1) / 2 <= grp[-1].y1 + 3):
                raise ValueError(f"PICO {q['n']}: label {lab.text!r} not beside its row")
            rows.append(f"{lab.text.strip()} | {_cell_text(grp, vocab)}")
        parts = [f"{q['n']} {join_lines(q['title'], vocab)}", *rows]
        if q["notes"]:
            parts.append(join_lines(q["notes"], vocab))
        out.append((q["n"], normalise_text("\n".join(parts)), min(q["pages"]), max(q["pages"])))
    return out


def _cell_text(group: list[Line], vocab: set[str]) -> str:
    """Text of one table cell.

    Lines 11 pt apart are a wrapped sentence and are joined with a space; lines
    14 pt apart are separate list items (e.g. outcomes) and keep their line break.
    """
    items: list[list[str]] = [[group[0].text]]
    for prev, line in zip(group, group[1:]):
        if line.y0 - prev.y0 < 12.5:
            items[-1].append(line.text)
        else:
            items.append([line.text])
    return "\n".join(join_lines(item, vocab) for item in items)


# --------------------------------------------------------------------------- #
# Main walk
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """A page element in reading order: a text line or a figure."""

    page: int
    y0: float
    line: Line | None = None
    figure: Figure | None = None


def read_items(doc: pymupdf.Document) -> list[Item]:
    hashes = type3_pua_glyph_hashes(doc)
    items: list[Item] = []
    for page in doc:
        spans = [s for s in extract_spans(page, hashes, GLYPH_TABLE) if not is_furniture(s)]
        text_spans, figures = split_figures(spans)
        page_items = [Item(page.number + 1, ln.y0, line=ln) for ln in group_lines(text_spans) if ln.text.strip()]
        page_items += [Item(page.number + 1, f.y0, figure=f) for f in figures]
        items += sorted(page_items, key=lambda it: it.y0)
    return items


def merge_heading_lines(items: list[Item]) -> list[Item]:
    """Merge wrapped heading lines into one line.

    Styled headings merge with an adjacent line of the same level on the same
    page. The black "Algorithm N: ..." headings in section 6.2 carry no styling,
    so they absorb the line directly below unless that line is a bullet.
    """
    out: list[Item] = []
    for it in items:
        if it.line is not None and out and out[-1].line is not None and it.page == out[-1].page:
            prev = out[-1].line
            gap = it.line.y0 - prev.y1
            lvl = is_heading(it.line)
            same_level = lvl and lvl == is_heading(prev) and gap < 12
            algorithm_wrap = (
                ALGORITHM_RE.match(prev.text.strip()) is not None
                and gap < 5
                and not BULLET_RE.match(it.line.text.strip())
            )
            if same_level or algorithm_wrap:
                prev.spans.extend(it.line.spans)
                continue
        out.append(it)
    return out


def parse_grading(grading: str) -> tuple[str, str]:
    """'Strong recommendation, moderate- to high-certainty evidence' -> ('strong', 'moderate to high')."""
    m = GRADING_RE.match(grading)
    if not m:
        raise ValueError(f"Unrecognised grading line: {grading!r}")
    strength = m.group(1).lower()
    cm = CERTAINTY_RE.match(m.group(2).strip())
    if not cm:
        raise ValueError(f"Unrecognised certainty: {m.group(2)!r}")
    certainty = f"{cm.group(1)} to {cm.group(2)}" if cm.group(1) else cm.group(2)
    return strength, certainty


def heading_text(line: Line) -> str:
    return normalise_text(re.sub(r"\s+", " ", line.text)).strip()


class Walker:
    """Walk items in reading order, building leaves, statements and table chunks."""

    def __init__(self, vocab: set[str], report: ParseReport) -> None:
        self.vocab = vocab
        self.report = report
        self.path: list[str] = []
        self.chapter: str = ""
        self.leaves: list[Leaf] = []
        self.order: list[object] = []  # Leaf | Statement | Chunk, in document order
        self.statements: list[Statement] = []
        self.exec_gradings: list[str] = []
        self.tables: list[Chunk] = []
        self._buffer: list[Line] = []
        self._leaf: Leaf | None = None
        self._rec: dict | None = None
        self._annex4_lines: list[Line] = []

    # -- leaf management ---------------------------------------------------- #
    def _flush_lines(self) -> None:
        if self._buffer and self._leaf is not None:
            self._leaf.blocks += lines_to_blocks(self._buffer, self.vocab)
        self._buffer = []

    def _close_leaf(self) -> None:
        self._flush_lines()
        if self._leaf is not None and self._leaf.blocks:
            self.leaves.append(self._leaf)
            self.order.append(self._leaf)
        self._leaf = None

    def _open_leaf(self, path: list[str], standalone: bool = False) -> None:
        self._close_leaf()
        self._leaf = Leaf(list(path), standalone=standalone, seq=len(self.statements))

    # -- recommendation boxes ------------------------------------------------ #
    def _start_rec(self, line: Line) -> None:
        self._close_leaf()
        m = REC_TITLE_RE.match(line.text.strip())
        assert m
        self._rec = {"n": int(m.group(1)), "title": [line.text.strip()], "lines": [], "in_title": True, "remarks": False}

    def _rec_line(self, line: Line) -> bool:
        """Consume a line inside a recommendation box. Return False when the box has ended."""
        rec = self._rec
        assert rec is not None
        text = line.text.strip()
        if rec["in_title"] and line.main_span.color == WHITE:
            rec["title"].append(text)
            return True
        rec["in_title"] = False
        inside = line.x0 >= body_margin(line.page) + 5
        if is_heading(line) or not inside:
            self._end_rec()
            return False
        if text.startswith("Implementation remarks"):
            rec["remarks"] = True
            self._open_leaf(self.path + ["Implementation remarks"], standalone=True)
            assert self._leaf is not None
            self._leaf.parent_rec = rec["n"]
            self._buffer.append(line)
            return True
        if rec["remarks"]:
            self._buffer.append(line)
            return True
        if GRADING_RE.match(text):
            stmt_lines: list[Line] = rec["lines"]
            blocks = lines_to_blocks(stmt_lines, self.vocab)
            title = normalise_text(" ".join(rec["title"]))
            rec["part"] = rec.get("part", 0) + 1
            stmt = Statement(
                rec_number=rec["n"],
                rec_part=rec["part"],
                title=title,
                text=normalise_text("\n".join(b.text for b in blocks)),
                grading=normalise_text(text),
                page_start=stmt_lines[0].page,
                page_end=line.page,
                section_path=list(self.path),
            )
            self.statements.append(stmt)
            self.order.append(stmt)
            rec["lines"] = []
            return True
        rec["lines"].append(line)
        return True

    def _end_rec(self) -> None:
        if self._rec is not None and self._rec["lines"]:
            raise ValueError(f"Recommendation {self._rec['n']}: statement without grading line")
        self._close_leaf()
        self._rec = None
        self._open_leaf(self.path)

    # -- figures ------------------------------------------------------------- #
    def _figure(self, fig: Figure) -> None:
        if fig.caption.startswith(SKIPPED_FIGURES):
            self.report.skipped.append(f"{fig.caption} (PDF p. {fig.page}): diagram; content given in text/Annex 4")
            return
        if fig.caption.startswith("Table 1"):
            self._table_chunk("table-1", fig.caption, certainty_table(fig, self.vocab), fig.page, fig.page)
            return
        if fig.caption.startswith("Fig. A3.1"):
            self._table_chunk("fig-a3-1", fig.caption, ratings_table(fig), fig.page, fig.page)
            return
        # Figs 3–6: flowcharts, kept in the running text of their section.
        self._flush_lines()
        if self._leaf is None:
            self._open_leaf(self.path)
        assert self._leaf is not None
        self._leaf.blocks.append(flowchart_block(fig, self.vocab))

    def _table_chunk(self, slug: str, caption: str, body: str, p0: int, p1: int) -> None:
        text = normalise_text(f"{caption}\n{body}")
        chunk = Chunk(f"{DOC_ID}-{slug}", self.path + [caption], "table", text, p0, p1)
        self.tables.append(chunk)
        self.order.append(chunk)

    # -- main entry ---------------------------------------------------------- #
    def walk(self, items: list[Item]) -> None:
        include = False
        in_exec = False
        for it in items:
            line = it.line
            if line is not None and is_heading(line) == 1:
                self._close_leaf()
                if self._rec is not None:
                    self._end_rec()
                    self._close_leaf()
                if self.chapter.startswith("Annex 4"):
                    self._emit_pico()
                title = heading_text(line)
                self.chapter = title
                include = title.startswith(INCLUDED_CHAPTER_PREFIXES)
                in_exec = title == EXEC_SUMMARY_TITLE
                self.path = [title]
                if include:
                    self._open_leaf(self.path)
                continue
            if in_exec:
                if line is not None and GRADING_RE.match(line.text.strip()):
                    self.exec_gradings.append(normalise_text(line.text.strip()))
                continue
            if not include:
                continue
            if it.figure is not None:
                self._figure(it.figure)
                continue
            assert line is not None
            if self.chapter.startswith("Annex 4") and (self._annex4_lines or (line.main_span.color == RED and line.main_span.size == 11.0)):
                self._annex4_lines.append(line)
                continue
            if self._rec is not None and self._rec_line(line):
                continue
            text = line.text.strip()
            if line.main_span.color == WHITE and REC_TITLE_RE.match(text):
                self._start_rec(line)
                continue
            level = is_heading(line)
            if level in (2, 3) or ALGORITHM_RE.match(text):
                level = level or 3
                title = heading_text(line)
                # A sub-heading hangs under the current section, or directly under the
                # chapter when the chapter has no numbered sections.
                parent_depth = 1 if level == 2 else min(len(self.path), 2)
                self.path = self.path[:parent_depth] + [title]
                self._open_leaf(self.path)
                continue
            if self._leaf is None:
                self._open_leaf(self.path)
            self._buffer.append(line)
        if self._rec is not None:
            self._end_rec()
        if self.chapter.startswith("Annex 4"):
            self._emit_pico()
        self._close_leaf()

    def _emit_pico(self) -> None:
        if not self._annex4_lines:
            return
        for n, text, p0, p1 in pico_tables(self._annex4_lines, self.vocab):
            chunk = Chunk(f"{DOC_ID}-annex-4-pico-{n:02d}", [self.chapter, f"PICO question {n}"], "pico", text, p0, p1)
            self.tables.append(chunk)
            self.order.append(chunk)
        self._annex4_lines = []


# --------------------------------------------------------------------------- #
# Leaves -> prose chunks
# --------------------------------------------------------------------------- #
def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:max_words])


def path_slug(path: list[str]) -> str:
    """'3 Recommendations' > '3.1 Blood ...' > 'Evidence and rationale' -> 's3-1-evidence-and-rationale'."""
    parts: list[str] = []
    for p in path:
        num = re.match(r"^(\d+(?:\.\d+)*)\s", p)
        annex = re.match(r"^Annex (\d+)", p)
        if annex:
            parts = [f"annex-{annex.group(1)}"]
        elif num:
            parts = ["s" + num.group(1).replace(".", "-")]  # a section number implies its parents
        else:
            parts.append(slugify(p))
    return "-".join(parts)


def section_number(path: list[str]) -> str | None:
    """Deepest numbered section in a path, e.g. '2-3' for [..., '2.3 Outcome ...']."""
    for p in reversed(path):
        num = re.match(r"^(\d+(?:\.\d+)*)\s", p)
        if num:
            return num.group(1).replace(".", "-")
    return None


def _words(group: list[Leaf]) -> int:
    return sum(leaf.words for leaf in group)


def _merge_target(groups: list[list[Leaf]], i: int) -> int | None:
    """Pick the group that a small group ``i`` should merge into, or None.

    Preference order:
    1. next / previous group in the same numbered section (path[:2]), skipping
       over standalone implementation remarks; a target that already exceeds
       MAX_WORDS is acceptable because it will be split anyway;
    2. next / previous group in the same chapter (path[:1]) with no
       recommendation box in between, if the result stays within MAX_WORDS.
    """
    grp = groups[i]
    size = _words(grp)

    def neighbours(step: int, key_len: int) -> int | None:
        j = i + step
        while 0 <= j < len(groups):
            other = groups[j]
            if other[0].path[:key_len] != grp[0].path[:key_len]:
                return None
            if not other[0].standalone:
                return j
            j += step
        return None

    for step in (1, -1):
        j = neighbours(step, 2)
        if j is not None and (size + _words(groups[j]) <= MAX_WORDS or _words(groups[j]) > MAX_WORDS):
            return j
    for step in (1, -1):
        j = neighbours(step, 1)
        if j is not None and groups[j][0].seq == grp[0].seq and size + _words(groups[j]) <= MAX_WORDS:
            return j
    return None


def merge_small_leaves(leaves: list[Leaf], report: ParseReport) -> dict[int, Leaf]:
    """Merge prose leaves under MIN_WORDS with a neighbour (see :func:`_merge_target`).

    Implementation remarks are never merged. A merged leaf takes the common
    path prefix of its parts, and the headings below that prefix are kept as
    lines of text so the chunk still reads in document order.

    Returns ``{id(first original leaf of a group): merged leaf}``; original leaves
    that were absorbed into a group led by another leaf have no entry.
    """
    groups: list[list[Leaf]] = [[leaf] for leaf in leaves]
    changed = True
    while changed:
        changed = False
        for i, grp in enumerate(groups):
            if grp[0].standalone or _words(grp) >= MIN_WORDS:
                continue
            j = _merge_target(groups, i)
            if j is None:
                continue
            lo, hi = min(i, j), max(i, j)
            groups[lo] = groups[lo] + groups[hi]
            del groups[hi]
            changed = True
            break

    merged: dict[int, Leaf] = {}
    for grp in groups:
        lead = grp[0]
        if len(grp) == 1:
            merged[id(lead)] = lead
            continue
        common = lead.path
        for leaf in grp[1:]:
            k = 0
            while k < min(len(common), len(leaf.path)) and common[k] == leaf.path[k]:
                k += 1
            common = common[:k]
        blocks: list[Block] = []
        shown: list[str] = list(common)
        for leaf in grp:
            page = leaf.blocks[0].page_start
            for depth in range(len(common), len(leaf.path)):
                if depth >= len(shown) or shown[depth] != leaf.path[depth]:
                    blocks.append(Block(leaf.path[depth], page, page))
            shown = list(leaf.path)
            blocks += leaf.blocks
        report.notes.append(f"merged {' + '.join(repr(l.path[-1]) for l in grp)} -> {common[-1]!r}")
        leaf = Leaf(common, blocks, seq=lead.seq)
        first_no, last_no = section_number(lead.path), section_number(grp[-1].path)
        if first_no and last_no and first_no != last_no and section_number(common) != first_no:
            # Spans several numbered sections: name it by the range, e.g. 's2-1-to-2-3'.
            leaf.slug = f"s{first_no}-to-{last_no}"
        merged[id(lead)] = leaf
    return merged


def split_leaf(leaf: Leaf) -> list[list[Block]]:
    """Split a leaf into balanced parts of at most MAX_WORDS, at block boundaries."""
    total = leaf.words
    if total <= MAX_WORDS:
        return [leaf.blocks]
    n = math.ceil(total / MAX_WORDS)
    target = total / n
    parts: list[list[Block]] = [[]]
    count = 0
    for block in leaf.blocks:
        if parts[-1] and count + block.words > MAX_WORDS or (count >= target and len(parts) < n):
            parts.append([])
            count = 0
        parts[-1].append(block)
        count += block.words
    return parts


def rec_id(number: int) -> str:
    """Recommendation label as used in ``rec_id`` / ``parent_rec_id``."""
    return f"Recommendation {number}"


def leaves_to_chunks(walker: Walker, report: ParseReport) -> list[Chunk]:
    """Turn the walker's ordered output into the final, ordered chunk list."""
    # Merged leaves are emitted at the position of their first constituent.
    first_of = merge_small_leaves(walker.leaves, report)

    chunks: list[Chunk] = []
    used_ids: Counter = Counter()
    stmt_index: dict[int, list[Statement]] = {}
    for s in walker.statements:
        stmt_index.setdefault(s.rec_number, []).append(s)

    for obj in walker.order:
        if isinstance(obj, Leaf):
            leaf = first_of.get(id(obj))
            if leaf is None:
                continue
            parts = split_leaf(leaf)
            base = f"{DOC_ID}-{leaf.slug or path_slug(leaf.path)}"
            for k, blocks in enumerate(parts, start=1):
                cid = base if len(parts) == 1 else f"{base}-p{k}"
                used_ids[cid] += 1
                if used_ids[cid] > 1:
                    cid = f"{cid}-{used_ids[cid]}"
                text = normalise_text("\n".join(b.text for b in blocks))
                chunks.append(
                    Chunk(
                        cid,
                        leaf.path,
                        "prose",
                        text,
                        min(b.page_start for b in blocks),
                        max(b.page_end for b in blocks),
                        parent_rec_id=rec_id(leaf.parent_rec) if leaf.parent_rec is not None else None,
                    )
                )
            if len(parts) > 1:
                report.notes.append(f"split {leaf.path[-1]!r} ({leaf.words} words) into {len(parts)} parts")
        elif isinstance(obj, Statement):
            group = stmt_index[obj.rec_number]
            suffix = "" if len(group) == 1 else "abcdefgh"[obj.rec_part - 1]
            chunks.append(
                Chunk(
                    chunk_id=f"{DOC_ID}-rec-{obj.rec_number:02d}{suffix}",
                    section_path=obj.section_path,
                    chunk_type="recommendation",
                    text=normalise_text(f"{obj.title}\n{obj.text}\n{obj.grading}"),
                    page_start=obj.page_start,
                    page_end=obj.page_end,
                    rec_id=rec_id(obj.rec_number),
                    rec_part=obj.rec_part,
                )
            )
        elif isinstance(obj, Chunk):
            chunks.append(obj)
    return chunks


def assign_grading(chunks: list[Chunk], walker: Walker, report: ParseReport) -> None:
    """Set strength/certainty on recommendation chunks.

    ``strength_as_printed`` and ``certainty`` are the values printed beneath the
    statement in section 3. Two cross-checks look for conflicts: the Executive
    summary's grading of the same statement, and the GRADE wording convention
    ("WHO recommends" = strong, "WHO suggests" = conditional).

    * No conflict: ``strength`` = printed value.
    * Conflict listed in ``GRADING_RESOLUTIONS`` (and the printed value still
      matches the one reviewed): ``strength`` = resolved value and
      ``source_discrepancy`` = the documented note.
    * Any other conflict: ``strength`` = None, reported as unresolved.
    """
    by_key = {(s.rec_number, s.rec_part): s for s in walker.statements}
    stmts = walker.statements
    if len(walker.exec_gradings) != len(stmts):
        report.conflicts.append(
            f"Executive summary has {len(walker.exec_gradings)} grading lines vs {len(stmts)} statements in section 3; "
            "cross-check skipped"
        )
        exec_by_stmt: dict[int, str] = {}
    else:
        exec_by_stmt = {id(s): g for s, g in zip(stmts, walker.exec_gradings)}
    for chunk in chunks:
        if chunk.chunk_type != "recommendation":
            continue
        key = (int(chunk.rec_id.split()[-1]), chunk.rec_part)  # type: ignore[union-attr, arg-type]
        stmt = by_key[key]
        printed, certainty = parse_grading(stmt.grading)
        chunk.strength_as_printed, chunk.certainty = printed, certainty

        strength_conflicts: list[str] = []
        eg = exec_by_stmt.get(id(stmt))
        if eg is not None:
            es, ec = parse_grading(eg)
            if es != printed:
                strength_conflicts.append(f"section 3 prints {printed!r}; Executive summary prints {es!r}")
            if ec != certainty:
                # No certainty conflicts exist in this document; fail loudly rather than guess.
                raise ValueError(f"{chunk.chunk_id}: unreviewed certainty conflict ({certainty!r} vs {ec!r})")
        verb = re.search(r"\bWHO (recommends|suggests)\b", stmt.text)
        if verb:
            implied = "strong" if verb.group(1) == "recommends" else "conditional"
            if implied != printed:
                strength_conflicts.append(f"wording 'WHO {verb.group(1)}' implies {implied!r}; printed {printed!r}")

        resolution = GRADING_RESOLUTIONS.get(key)
        if resolution is not None and resolution.printed != printed:
            raise ValueError(f"{chunk.chunk_id}: resolution expects printed {resolution.printed!r}, found {printed!r}")
        if not strength_conflicts:
            if resolution is not None:
                raise ValueError(f"{chunk.chunk_id}: resolution listed but no conflict found; remove it")
            chunk.strength = printed
        elif resolution is not None:
            chunk.strength = resolution.resolved
            chunk.source_discrepancy = resolution.note
            report.conflicts.append(f"{chunk.chunk_id}: RESOLVED as {resolution.resolved!r} ({'; '.join(strength_conflicts)})")
        else:
            chunk.strength = None
            report.conflicts.append(f"{chunk.chunk_id}: UNRESOLVED, strength set to null ({'; '.join(strength_conflicts)})")


def assign_printed_pages(chunks: list[Chunk], numbers: dict[int, int | None], roman: set[int]) -> None:
    """Copy each chunk's printed page numbers from the page-number tabs of its PDF pages."""
    for chunk in chunks:
        for pno in (chunk.page_start, chunk.page_end):
            if pno in roman:
                raise ValueError(f"{chunk.chunk_id}: PDF page {pno} has a roman page number; front matter is not chunked")
        chunk.printed_page_start = numbers.get(chunk.page_start)
        chunk.printed_page_end = numbers.get(chunk.page_end)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def parse(pdf_path: Path | None = None) -> tuple[list[Chunk], ParseReport]:
    """Parse the guideline PDF and return ordered chunks plus a report."""
    pdf_path = pdf_path or get_source("hypertension_pdf").path
    doc = pymupdf.open(pdf_path)
    report = ParseReport()
    items = merge_heading_lines(read_items(doc))
    vocab = build_vocabulary(it.line.text for it in items if it.line is not None)
    walker = Walker(vocab, report)
    walker.walk(items)
    chunks = leaves_to_chunks(walker, report)
    assign_grading(chunks, walker, report)
    assign_printed_pages(chunks, *printed_page_numbers(doc))
    ids = [c.chunk_id for c in chunks]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"Duplicate chunk ids: {dupes}")
    return chunks, report


def write_jsonl(chunks: list[Chunk], path: Path = OUTPUT_PATH) -> None:
    """Write chunks as UTF-8 JSON Lines with LF line endings (byte-for-byte reproducible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")


def print_summary(chunks: list[Chunk], report: ParseReport) -> None:
    by_type = Counter(c.chunk_type for c in chunks)
    words = [len(c.text.split()) for c in chunks]
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[2])}")
    print("Chunks by type: " + ", ".join(f"{t}={n}" for t, n in sorted(by_type.items())))
    print(f"Words per chunk: min={min(words)} median={statistics.median(words):g} max={max(words)}")
    for t in sorted(by_type):
        w = [len(c.text.split()) for c in chunks if c.chunk_type == t]
        print(f"  {t:<14} min={min(w)} median={statistics.median(w):g} max={max(w)}")
    print("Recommendation chunks:")
    for c in chunks:
        if c.chunk_type == "recommendation":
            print(
                f"  {c.chunk_id:<24} {c.rec_id:<18} part={c.rec_part} strength={c.strength!s:<12} "
                f"certainty={c.certainty!s:<16} pdf p.{c.page_start} / printed p.{c.printed_page_start}"
            )
    rec_ids = sorted({c.rec_id for c in chunks if c.rec_id}, key=lambda r: int(r.split()[-1]))
    print("rec_ids found: " + ", ".join(rec_ids))
    remarks = [f"{c.chunk_id} -> {c.parent_rec_id}" for c in chunks if c.parent_rec_id]
    print("Implementation remarks linked: " + "; ".join(remarks))
    print("Grading conflicts in the source:")
    for a in report.conflicts or ["none"]:
        print(f"  - {a}")
    print("Skipped content:")
    for s in report.skipped:
        print(f"  - {s}")
    print("Merge/split decisions:")
    for n in report.notes:
        print(f"  - {n}")


def main() -> None:
    chunks, report = parse()
    write_jsonl(chunks)
    print_summary(chunks, report)


if __name__ == "__main__":
    main()
