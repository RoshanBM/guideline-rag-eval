"""Which chunks are relevant to an eval item: the gold-span rule used by the retrieval metrics.

Relevance is defined by the gold spans, not by chunk ids, so it survives any change of
chunking: a chunk is relevant to an item when it comes from the same document as one of the
item's gold sources, covers that source's printed page, and its verbatim text contains the
source's span. Matching uses ``text_utils.normalise_for_match`` on both sides (whitespace,
quote and dash styles folded, documented typo corrections treated as equivalent; case kept).

"Verbatim text" is the chunk's ``source_text`` plus its footnotes (``schema.verifiable_text``),
the same text quotes are verified against: footnotes are source text too (e.g. table notes).
``context_header``, ``flow_text``, ``aware_category_note`` and the corrected retrieval ``text``
never count. (Footnotes are stored only inside ``text``, where the documented typo corrections
have been applied; because ``normalise_for_match`` applies the same corrections to both sides,
a span matches a corrected footnote exactly when it matches the printed one.)

Relevance is decided **at chunk level**: a chunk covering printed pages 18–19 is relevant to a
span cited on page 19 even if the span is printed on page 18. Pinning a span to its exact page
needs the raw PDF and is done only when the eval set is validated (``scripts/validate_eval.py``,
page check), so that retrieval metrics depend on nothing but the chunk files.
``validate_eval.py`` applies this same chunk rule first, so every validated span has at least
one relevant chunk.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.evaluate.eval_schema import EvalItem, GoldSource, parse_item
from src.parse.schema import verifiable_text
from src.parse.sources import CHUNKS_DIR
from src.parse.text_utils import normalise_for_match


def load_chunks(chunks_dir: Path = CHUNKS_DIR) -> list[dict[str, Any]]:
    """Every chunk record from ``chunks_dir/*.jsonl``, files in name order, lines in file order."""
    chunks: list[dict[str, Any]] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            chunks.extend(json.loads(line) for line in fh if line.strip())
    return chunks


def chunk_match_text(chunk: dict[str, Any]) -> str:
    """The chunk's verbatim text (source text plus footnotes), match-normalised."""
    return normalise_for_match(verifiable_text(chunk))


def covers_page(chunk: dict[str, Any], printed_page: int) -> bool:
    start, end = chunk.get("printed_page_start"), chunk.get("printed_page_end")
    return start is not None and end is not None and start <= printed_page <= end


def source_matches(source: GoldSource, chunk: dict[str, Any]) -> bool:
    """True if ``chunk`` holds ``source``: same document, page covered, span contained."""
    return (
        chunk.get("doc_id") == source.doc_id
        and covers_page(chunk, source.printed_page)
        and normalise_for_match(source.span) in chunk_match_text(chunk)
    )


def relevant_chunk_ids_by_source(
    item: EvalItem | dict[str, Any], chunks: Iterable[dict[str, Any]]
) -> list[list[str]]:
    """For each gold source of ``item`` (in order), the chunk ids (in ``chunks`` order) holding it.

    Used for per-source recall: a source counts as found when any of its chunks is retrieved.
    A span printed in two chunks (e.g. a footnote shared by the adult and child dosing tables)
    gives a source with two chunk ids; two sources can also share a chunk.
    """
    parsed = item if isinstance(item, EvalItem) else parse_item(item)
    chunks = list(chunks)
    return [[c["chunk_id"] for c in chunks if source_matches(src, c)] for src in parsed.gold_sources]


def relevant_chunk_ids(item: EvalItem | dict[str, Any], chunks: Iterable[dict[str, Any]]) -> list[str]:
    """Chunk ids (in ``chunks`` order, without duplicates) that contain any gold span of ``item``.

    ``item`` may be an :class:`EvalItem` or a raw item dict. Unanswerable items have no gold
    sources and therefore no relevant chunks.
    """
    parsed = item if isinstance(item, EvalItem) else parse_item(item)
    ids: list[str] = []
    for chunk in chunks:
        if chunk["chunk_id"] not in ids and any(source_matches(src, chunk) for src in parsed.gold_sources):
            ids.append(chunk["chunk_id"])
    return ids
