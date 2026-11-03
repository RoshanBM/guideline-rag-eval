"""Find verbatim source spans for eval items.

Usage::

    python scripts/find_span.py "cefalexin 375 mg" [--doc who-aware-2022] [--max 20]

Searches the verbatim text of every chunk in ``data/chunks/*.jsonl`` (``source_text`` plus
footnotes, as in ``src/evaluate/relevance.py``) case-insensitively, after
``text_utils.normalise_for_match`` on both sides (so whitespace, quote and dash styles and
the documented typo corrections do not matter). For each matching chunk it prints the
doc_id, chunk_id and printed page(s), and the smallest verbatim unit containing the query,
ready to paste as a gold ``span``: the sentence, else the table cell, else the line, else the
run of lines. A ready-made ``gold_sources`` entry is printed too; when the chunk covers two
printed pages, check which one the span is on before using it.

If the query occurs nowhere as a phrase, the search falls back to **all words**: chunks
containing every query term (a number keeps its unit: "cefalexin 375 mg" is the terms
"cefalexin" and "375 mg"), each shown with the shortest run of lines that contains all terms.
AWaRe dosing tables put the drug and its weight-band doses on separate lines, so this is how
"drug + dose" queries find them. Such results are labelled ``all words``; read the run to
check the dose belongs to that drug before using it as a span.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.relevance import load_chunks  # noqa: E402
from src.parse.schema import verifiable_text  # noqa: E402
from src.parse.text_utils import normalise_for_match  # noqa: E402

CHUNKS_DIR = ROOT / "data" / "chunks"
CELL_SEPARATOR = " | "  # parser-added between table cells
# A sentence ends at . ! ? or ; followed by a space and a capital letter or a bullet. Lower-case
# continuations ("e.g. walking") and numbers ("Fig. 1") do not end a sentence.
SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z•])|\s+(?=• )")


# A number (or range) with its unit is one term; anything else splits on whitespace.
TERM_RE = re.compile(r"\d[\d.,–-]*\s*(?:mg/kg/dose|mg/kg/day|mg/kg|mg|mcg|g|mL|ml|L|units?|IU|mmol/L|%)(?![A-Za-z])|\S+")


def _key(text: str) -> str:
    return normalise_for_match(text).lower()


def terms(query: str) -> list[str]:
    return [_key(t) for t in TERM_RE.findall(query)]


def _has_all(text: str, query_terms: list[str]) -> bool:
    key = _key(text)
    return all(re.search(rf"(?<![\w.]){re.escape(t)}(?!\w)", key) for t in query_terms)


def all_words_span(text: str, query: str) -> str | None:
    """The shortest run of consecutive lines of ``text`` containing every query term."""
    query_terms = terms(query)
    if not query_terms or not _has_all(text, query_terms):
        return None
    lines = text.split("\n")
    best: list[str] | None = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines) + 1):
            if _has_all(" ".join(lines[i:j]), query_terms):
                if best is None or j - i < len(best):
                    best = lines[i:j]
                break
    if not best:
        return None
    # Start at the first table cell that is needed (drop leading cells such as a row label).
    cells = best[0].split(CELL_SEPARATOR)
    while len(cells) > 1 and _has_all(" ".join([CELL_SEPARATOR.join(cells[1:]), *best[1:]]), query_terms):
        cells = cells[1:]
    return clean_span(" ".join([CELL_SEPARATOR.join(cells), *best[1:]]))


def sentences(cell: str) -> list[str]:
    return [s for s in SENTENCE_END_RE.split(cell) if s.strip()]


def clean_span(text: str) -> str:
    """Strip the parser's bullet marker, which is formatting rather than source text."""
    return re.sub(r"^•\s*", "", text.strip())


def best_span(text: str, query: str) -> str | None:
    """The smallest verbatim unit of ``text`` that contains ``query`` (see module docstring)."""
    needle = _key(query)
    if needle not in _key(text):
        return None
    lines = text.split("\n")
    for line in lines:
        for cell in line.split(CELL_SEPARATOR):
            for sentence in sentences(cell):
                if needle in _key(sentence):
                    return clean_span(sentence)
    for line in lines:
        for cell in line.split(CELL_SEPARATOR):
            if needle in _key(cell):
                return clean_span(cell)
    for line in lines:
        if needle in _key(line):
            return clean_span(line)
    # The query crosses line breaks: the shortest run of consecutive lines containing it.
    best: list[str] | None = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines) + 1):
            run = lines[i:j]
            if needle in _key(" ".join(run)):
                if best is None or len(run) < len(best):
                    best = run
                break
    return clean_span(" ".join(best)) if best else None


def find(query: str, chunks: list[dict[str, Any]], doc_id: str | None = None) -> list[dict[str, Any]]:
    """One result per matching chunk: doc_id, chunk_id, pages, the span to paste and the match
    type (``phrase``, or ``all words`` when the phrase occurs in no chunk)."""
    selected = [c for c in chunks if not doc_id or c["doc_id"] == doc_id]
    for match, finder in (("phrase", best_span), ("all words", all_words_span)):
        results = []
        for c in selected:
            span = finder(verifiable_text(c), query)
            if span is None:
                continue
            results.append({
                "doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "match": match,
                "printed_page_start": c["printed_page_start"], "printed_page_end": c["printed_page_end"],
                "pdf_page_start": c["pdf_page_start"], "pdf_page_end": c["pdf_page_end"], "span": span,
            })
        if results:
            return results
    return []


def _pages(start: int | None, end: int | None, prefix: str = "") -> str:
    return f"{prefix}p. {start}" if start == end else f"{prefix}pp. {start}–{end}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query", help="text to look for, e.g. \"cefalexin 375 mg\"")
    parser.add_argument("--doc", help="restrict to one doc_id")
    parser.add_argument("--max", type=int, default=20, help="maximum number of matches to print (default 20)")
    parser.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    args = parser.parse_args(argv)

    results = find(args.query, load_chunks(args.chunks_dir), args.doc)
    match = results[0]["match"] if results else "phrase"
    print(f"{len(results)} matching chunk(s) for {args.query!r}" + (f" in {args.doc}" if args.doc else "")
          + (" (no phrase match; all words, shortest run of lines)" if match == "all words" else ""))
    for r in results[: args.max]:
        printed = _pages(r["printed_page_start"], r["printed_page_end"])
        pdf = _pages(r["pdf_page_start"], r["pdf_page_end"], prefix="PDF ")
        print(f"\n{r['doc_id']} | {r['chunk_id']} | printed {printed} ({pdf})")
        print(f"  span: {r['span']}")
        source = {"doc_id": r["doc_id"], "printed_page": r["printed_page_start"], "span": r["span"]}
        note = "" if r["printed_page_start"] == r["printed_page_end"] else \
            f"   <- chunk covers {printed}: check the page"
        print(f"  gold_source: {json.dumps(source, ensure_ascii=False)}{note}")
    if len(results) > args.max:
        print(f"\n... {len(results) - args.max} more (use --max)")
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
