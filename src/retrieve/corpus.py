"""The retrieval corpus: the chunks in ``data/chunks/`` that are indexed by default.

Only chunks with ``index_default: true`` are indexed (so the hypertension PICO tables are
excluded). The indexed text is the chunk's retrieval ``text`` field (``context_header``,
``flow_text`` or ``source_text``, footnotes, ``aware_category_note``, typo corrections applied),
not ``source_text``. Each indexed chunk keeps its ``chunk_id``, ``doc_id``, ``collection`` and
``chunk_type``.

Run ``python -m src.retrieve.corpus`` to print the counts per document.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.evaluate.relevance import load_chunks
from src.parse.sources import CHUNKS_DIR


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    doc_id: str
    collection: str
    chunk_type: str
    text: str  # the chunk's retrieval text (record["text"])
    context_header: str | None = None  # parser-written header; when set, ``text`` starts with it

    def header_and_body(self) -> tuple[str, str]:
        """``(context_header, rest of text)``; ``("", text)`` when the chunk has no header."""
        if self.context_header and self.text.startswith(self.context_header + "\n"):
            return self.context_header, self.text[len(self.context_header) + 1:]
        if self.context_header:
            raise ValueError(f"{self.chunk_id}: text does not start with its context_header")
        return "", self.text


def load_corpus(chunks_dir: Path = CHUNKS_DIR) -> list[CorpusChunk]:
    """Indexed chunks (``index_default`` true) from ``chunks_dir/*.jsonl``, in file and line order."""
    corpus: list[CorpusChunk] = []
    seen: set[str] = set()
    for rec in load_chunks(chunks_dir):
        if rec["chunk_id"] in seen:
            raise ValueError(f"duplicate chunk_id {rec['chunk_id']!r} in {chunks_dir}")
        seen.add(rec["chunk_id"])
        if rec["index_default"] is not True:
            continue
        corpus.append(CorpusChunk(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            collection=rec["collection"],
            chunk_type=rec["chunk_type"],
            text=rec["text"],
            context_header=rec["context_header"],
        ))
    if not corpus:
        raise ValueError(f"no indexable chunks found in {chunks_dir}")
    return corpus


def corpus_fingerprint(corpus: list[CorpusChunk]) -> str:
    """SHA-256 over the ids and texts, in order: detects a saved index built from other chunks."""
    h = hashlib.sha256()
    for chunk in corpus:
        h.update(chunk.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(chunk.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


def count_table(chunks_dir: Path = CHUNKS_DIR) -> list[tuple[str, str, int, int]]:
    """``(doc_id, collection, n_indexed, n_excluded)`` per document, in first-seen order."""
    indexed: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    collection: dict[str, str] = {}
    for rec in load_chunks(chunks_dir):
        collection.setdefault(rec["doc_id"], rec["collection"])
        (indexed if rec["index_default"] is True else excluded)[rec["doc_id"]] += 1
    return [(doc, collection[doc], indexed[doc], excluded[doc]) for doc in collection]


def print_counts(chunks_dir: Path = CHUNKS_DIR) -> None:
    rows = count_table(chunks_dir)
    print(f"{'doc_id':<22} {'collection':<16} {'indexed':>7} {'excluded':>8}")
    for doc, coll, n_in, n_out in rows:
        print(f"{doc:<22} {coll:<16} {n_in:>7} {n_out:>8}")
    print(f"{'total':<22} {'':<16} {sum(r[2] for r in rows):>7} {sum(r[3] for r in rows):>8}")
    by_type = Counter(c.chunk_type for c in load_corpus(chunks_dir))
    print("indexed by chunk_type: " + ", ".join(f"{t} {n}" for t, n in sorted(by_type.items())))


if __name__ == "__main__":
    print_counts()
