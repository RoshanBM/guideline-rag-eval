"""Validate ``eval/questions.jsonl`` against the eval schema, the chunk files and the raw PDFs.

Usage::

    python scripts/validate_eval.py [--questions eval/questions.jsonl] [--chunks-dir data/chunks]
                                    [--strict] [--skip-page-check]

Checks, for every line:

* it is a JSON object that passes ``src/evaluate/eval_schema.validate_item``;
* its ``qid`` is unique;
* its ``collection`` (if set) exists in the chunk files, and each gold source's ``doc_id``
  exists and belongs to that collection;
* **chunk check**: each gold span occurs verbatim (after ``text_utils.normalise_for_match``)
  in the source text or footnotes of at least one chunk of that ``doc_id`` whose printed-page
  range covers the cited ``printed_page``. This is the relevance rule of
  ``src/evaluate/relevance.py``; parser-written text (``context_header``, ``flow_text``,
  ``aware_category_note``) never counts. On failure the report says where the span does
  occur, if anywhere. A span matching more than one chunk is a warning listing their ids, so
  a more specific span can be chosen.
* **page check** (this script only): the span also occurs in the raw text of the exact cited
  page of the source PDF, read with PyMuPDF. Chunks can cover two pages, so only this check
  pins a span to its page. The printed page is mapped to the PDF page by the same mapping the
  parser used (page labels for AWaRe, footer numbers for HEARTS-D and hypertension). A
  mismatch is an error. See :func:`page_key` and :func:`on_page` for how raw text is compared.

It then prints per-category counts against ``TARGETS``. Shortfalls are reported but are not
errors unless ``--strict`` is given (useful once the question set is meant to be complete).
``--skip-page-check`` turns the page check off (e.g. where the raw PDFs are not available).
Exit status: 0 if valid, 1 if any error (or, with ``--strict``, any count off target).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.eval_schema import CATEGORIES, EvalItem, GoldSource, parse_item, validate_item  # noqa: E402
from src.evaluate.relevance import chunk_match_text, load_chunks, source_matches  # noqa: E402
from src.parse import who_hearts_d, who_hypertension  # noqa: E402
from src.parse.pdf_utils import PUA_RE, _font_xref, page_label_map, type3_pua_glyph_hashes  # noqa: E402
from src.parse.sources import get_source  # noqa: E402
from src.parse.text_utils import normalise_for_match  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
CHUNKS_DIR = ROOT / "data" / "chunks"

TARGETS: dict[str, int] = {
    "single_rec": 10,
    "combined": 8,
    "dose_lookup": 12,
    "population_variant": 6,
    "no_antibiotic": 4,
    "conflicting_source": 3,
    "near_miss": 8,
    "out_of_domain": 4,
    "personal_advice": 4,
}
TARGET_TOTAL = 59
SPARE = 1
MIN_SPAN_WORDS = 3  # shorter spans are allowed but flagged: they may match by accident

assert list(TARGETS) == list(CATEGORIES) and sum(TARGETS.values()) == TARGET_TOTAL


# --------------------------------------------------------------------------- #
# Page check: raw text of the cited PDF page
# --------------------------------------------------------------------------- #


def _aware_pages(doc: pymupdf.Document) -> dict[int, int | None]:
    """PDF page -> printed page from the PDF's page labels (as checked in source inspection)."""
    return {pdf: int(label) for label, pdf in page_label_map(doc).items() if label.isdigit()}


@dataclass(frozen=True)
class RawSource:
    source_id: str  # key of src.parse.sources.SOURCES
    printed_pages: Callable[[pymupdf.Document], dict[int, int | None]]  # PDF page -> printed page
    glyph_table: dict[str, str]  # Private Use Area glyphs (browser-printed PDFs), see pdf_utils


# Raw PDF of each document, and the parser's own printed-page mapping. The AWaRe chunks come
# from the EPUB; its printed page numbers are the PDF's page labels (notes/source_inspection.md).
RAW_SOURCES: dict[str, RawSource] = {
    "who-htn-2021": RawSource("hypertension_pdf", lambda d: who_hypertension.printed_page_numbers(d)[0],
                              who_hypertension.GLYPH_TABLE),
    "who-hearts-d-2020": RawSource("hearts_d_pdf", who_hearts_d.printed_page_numbers, {}),
    "who-aware-2022": RawSource("aware_pdf", _aware_pages, {}),
}


def raw_page_text(page: pymupdf.Page, hashes: dict[tuple[int, str], str], glyph_table: dict[str, str]) -> str:
    """Raw text of one page, in PyMuPDF's block order.

    Private Use Area glyphs are resolved with ``glyph_table`` where known and dropped otherwise
    (in the AWaRe PDF they are icons). Superscript footnote markers are dropped (spans of at most
    3 characters, smaller than 0.8 of the line's largest span, not at the start of the line):
    the parsers remove them from the chunk text, and a gold span is copied from the chunk text.
    A marker that opens a line (the label of a footnote, e.g. "a Not for children") is kept.
    """
    out: list[str] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            spans = [(s, "".join(c["c"] for c in s["chars"])) for s in line["spans"]]
            largest = max((s["size"] for s, t in spans if t.strip()), default=0.0)
            seen_text = False
            for s, text in spans:
                if text.strip():
                    if seen_text and s["size"] < 0.8 * largest and len(text.strip()) <= 3:
                        continue
                    seen_text = True
                fx = _font_xref(s["font"])
                for c in s["chars"]:
                    ch = c["c"]
                    if PUA_RE.fullmatch(ch):
                        digest = hashes.get((fx, ch)) if fx is not None else None
                        out.append(glyph_table.get(digest, "") if digest else "")
                    else:
                        out.append(ch)
            out.append("\n")
    return "".join(out)


def page_key(text: str) -> str:
    """Comparison form for the page check.

    ``normalise_for_match`` (whitespace, quote and dash styles, typo corrections), lower case,
    then only letters, digits, ``% < > ≤ ≥ +`` and a decimal point or dash *between digits* are
    kept. Raw page text breaks lines and words ("signi-\\nficant", "cost-\\neffective") and reads
    some table cells in another order of spaces and punctuation; ignoring spaces and punctuation
    makes the check robust to that while still requiring every letter and number of the span, in
    order. Exponents ``²``/``³`` are dropped on both sides (the raw superscript is dropped).
    """
    t = normalise_for_match(text).lower().replace("²", "").replace("³", "")
    t = re.sub(r"(\d)\s*([.\-])\s*(\d)", r"\1\2\3", t)
    t = re.sub(r"(?<!\d)[.\-]|[.\-](?!\d)", "", t)
    return re.sub(r"[^0-9a-z%<>≤≥+.\-]", "", t)


PAGE_EDGE = 600  # key characters at the end/start of a page searched for a span crossing pages
MIN_PART = 8  # each part of a page-crossing span must have at least this many key characters


def on_page(span_key: str, page: str, next_page: str | None) -> bool:
    """True if the span is on ``page``, or starts near its end and finishes near the start of
    ``next_page`` (a sentence running over the page break; footers and running headers lie
    between, hence the ``PAGE_EDGE`` windows)."""
    if span_key in page:
        return True
    if next_page is None:
        return False
    tail, head = page[-PAGE_EDGE:], next_page[:PAGE_EDGE]
    return any(span_key[:cut] in tail and span_key[cut:] in head
               for cut in range(MIN_PART, len(span_key) - MIN_PART + 1))


class PageChecker:
    """Loads each raw PDF once and checks spans against the cited page."""

    def __init__(self) -> None:
        self._docs: dict[str, tuple[pymupdf.Document, dict[int, list[int]], dict[tuple[int, str], str]]] = {}
        self._keys: dict[tuple[str, int], str] = {}

    def _load(self, doc_id: str) -> tuple[pymupdf.Document, dict[int, list[int]], dict[tuple[int, str], str]]:
        if doc_id not in self._docs:
            raw = RAW_SOURCES[doc_id]
            doc = pymupdf.open(get_source(raw.source_id).path)
            to_pdf: dict[int, list[int]] = {}
            for pdf, printed in raw.printed_pages(doc).items():
                if printed is not None:
                    to_pdf.setdefault(printed, []).append(pdf)
            self._docs[doc_id] = (doc, to_pdf, type3_pua_glyph_hashes(doc))
        return self._docs[doc_id]

    def _key(self, doc_id: str, pdf_page: int) -> str | None:
        doc, _, hashes = self._load(doc_id)
        if not 1 <= pdf_page <= doc.page_count:
            return None
        if (doc_id, pdf_page) not in self._keys:
            text = raw_page_text(doc[pdf_page - 1], hashes, RAW_SOURCES[doc_id].glyph_table)
            self._keys[(doc_id, pdf_page)] = page_key(text)
        return self._keys[(doc_id, pdf_page)]

    def check(self, source: GoldSource) -> str | None:
        """None if the span is on the cited page, else the reason it is not."""
        if source.doc_id not in RAW_SOURCES:
            return f"no raw PDF registered for {source.doc_id} (add it to RAW_SOURCES)"
        try:
            _, to_pdf, _ = self._load(source.doc_id)
        except FileNotFoundError as exc:
            return f"raw PDF not available ({exc}); use --skip-page-check"
        pdf_pages = to_pdf.get(source.printed_page, [])
        if len(pdf_pages) != 1:
            return (f"printed page {source.printed_page} maps to PDF pages {pdf_pages}" if pdf_pages
                    else f"printed page {source.printed_page} not found in the PDF's page numbering")
        pdf = pdf_pages[0]
        span_key = page_key(source.span)
        if on_page(span_key, self._key(source.doc_id, pdf) or "", self._key(source.doc_id, pdf + 1)):
            return None
        _, to_pdf, _ = self._load(source.doc_id)
        found = [printed for printed, pages in sorted(to_pdf.items())
                 if any(span_key in (self._key(source.doc_id, p) or "") for p in pages)]
        where = f"; it is on printed page(s) {found}" if found else ""
        return f"span not in the raw text of printed page {source.printed_page} (PDF p. {pdf}){where}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass
class Report:
    items: list[EvalItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def counts(self) -> Counter:
        return Counter(item.category for item in self.items)

    def off_target(self) -> list[str]:
        """Category or total counts that miss the targets (used by ``--strict``)."""
        problems = [f"{cat}: {self.counts.get(cat, 0)} < {n}" for cat, n in TARGETS.items()
                    if self.counts.get(cat, 0) < n]
        total = len(self.items)
        if not TARGET_TOTAL <= total <= TARGET_TOTAL + SPARE:
            problems.append(f"total: {total} not in {TARGET_TOTAL}–{TARGET_TOTAL + SPARE}")
        return problems


def where_span_occurs(span: str, chunks: list[dict[str, Any]]) -> list[str]:
    """``doc_id p.N`` labels of chunks containing ``span`` (any document, any page)."""
    needle = normalise_for_match(span)
    found = []
    for c in chunks:
        if needle in chunk_match_text(c):
            pages = (f"p. {c['printed_page_start']}" if c["printed_page_start"] == c["printed_page_end"]
                     else f"pp. {c['printed_page_start']}–{c['printed_page_end']}")
            label = f"{c['doc_id']} {pages}"
            if label not in found:
                found.append(label)
    return found


def validate_file(path: Path, chunks: list[dict[str, Any]], page_check: bool = True) -> Report:
    report = Report()
    checker = PageChecker() if page_check else None
    collections = {c["collection"] for c in chunks}
    doc_collection = {c["doc_id"]: c["collection"] for c in chunks}
    seen: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not path.exists():
        report.errors.append(f"{path} does not exist")
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = f"line {lineno}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            report.errors.append(f"{where}: invalid JSON ({exc.msg})")
            continue
        qid = raw.get("qid") if isinstance(raw, dict) else None
        where = f"line {lineno} ({qid})" if qid else where
        problems = validate_item(raw)
        if problems:
            report.errors.extend(f"{where}: {p}" for p in problems)
            continue
        item = parse_item(raw)
        if item.qid in seen:
            report.errors.append(f"{where}: duplicate qid")
        seen.add(item.qid)
        if item.collection is not None and item.collection not in collections:
            report.errors.append(f"{where}: unknown collection {item.collection!r} (known: {sorted(collections)})")
        for i, src in enumerate(item.gold_sources):
            label = f"{where} gold_sources[{i}]"
            if src.doc_id not in doc_collection:
                report.errors.append(f"{label}: unknown doc_id {src.doc_id!r} (known: {sorted(doc_collection)})")
                continue
            if item.collection is not None and doc_collection[src.doc_id] != item.collection:
                report.errors.append(f"{label}: {src.doc_id} belongs to collection {doc_collection[src.doc_id]!r}, "
                                     f"not {item.collection!r}")
            matching = [c["chunk_id"] for c in chunks if source_matches(src, c)]
            if not matching:
                elsewhere = where_span_occurs(src.span, chunks)
                hint = f"; it occurs in {', '.join(elsewhere)}" if elsewhere else "; it occurs in no chunk"
                report.errors.append(f"{label}: span not found in {src.doc_id} on printed page "
                                     f"{src.printed_page}{hint}: {src.span!r}")
                continue
            if len(matching) > 1:
                report.warnings.append(f"{label}: span matches {len(matching)} chunks ({', '.join(matching)}); "
                                       f"a more specific span would identify one: {src.span!r}")
            if len(src.span.split()) < MIN_SPAN_WORDS:
                report.warnings.append(f"{label}: span has fewer than {MIN_SPAN_WORDS} words: {src.span!r}")
            if checker is not None:
                problem = checker.check(src)
                if problem:
                    report.errors.append(f"{label}: {problem}: {src.span!r}")
        report.items.append(item)
    return report


def print_report(path: Path, report: Report, page_check: bool = True) -> None:
    print(f"{path}: {len(report.items)} valid item(s), {len(report.errors)} error(s), "
          f"{len(report.warnings)} warning(s)" + ("" if page_check else " [page check skipped]"))
    for e in report.errors:
        print(f"  ERROR   {e}")
    for w in report.warnings:
        print(f"  WARNING {w}")
    counts = report.counts
    print("\ncategory             count  target")
    for cat, target in TARGETS.items():
        n = counts.get(cat, 0)
        mark = "" if n == target else (f"  ({target - n} to go)" if n < target else f"  (+{n - target})")
        print(f"{cat:<20} {n:>5}  {target:>6}{mark}")
    total = len(report.items)
    print(f"{'total':<20} {total:>5}  {TARGET_TOTAL:>6}  (+{SPARE} spare allowed)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    parser.add_argument("--strict", action="store_true", help="also fail when counts miss the targets")
    parser.add_argument("--skip-page-check", action="store_true", help="do not check spans against the raw PDFs")
    args = parser.parse_args(argv)
    page_check = not args.skip_page_check
    report = validate_file(args.questions, load_chunks(args.chunks_dir), page_check)
    print_report(args.questions, report, page_check)
    failed = bool(report.errors)
    if args.strict and report.off_target():
        print("\nOff target: " + "; ".join(report.off_target()))
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
