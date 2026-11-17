"""Retrieval metrics with binary relevance: hit@k, recall@k (per gold source), MRR@10, nDCG@k.

Every function takes a ranked list of chunk ids (best first, no duplicates) and the item's gold
sources as a sequence of chunk-id sets, one set per gold source: the chunks that hold that
source's span under the same-document, same-page rule of ``src/evaluate/relevance.py``
(:func:`~src.evaluate.relevance.relevant_chunk_ids_by_source`). The item's *relevant chunks*
are the union of those sets. Ranks are 1-based.

* ``recall@k`` (per gold source): a source is *found* at k if any chunk holding its span is in
  the top k; recall@k = found sources / gold sources. A combined question citing two documents
  gets 0.5 when only one is retrieved; a span printed in two chunks (e.g. a footnote shared by
  the adult and child dosing tables) is found by either.
* ``hit@k``: 1 if at least one relevant chunk is in the top k, else 0.
* ``mrr@10``: 1 / rank of the first relevant chunk if it is in the top 10, else 0.
* ``ndcg@k``: over relevant **chunks**: DCG@k / IDCG@k, gain 1 for a relevant chunk at rank r,
  discount 1 / log2(r + 1); IDCG@k is the DCG of min(k, |relevant chunks|) relevant chunks at
  the top. (So, unlike recall, nDCG still rewards retrieving every chunk holding a span.)

Items without gold sources (unanswerable items) are not scored: an empty source list, or a
source with no chunk, raises ``ValueError``; the caller excludes unanswerable items.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

KS: tuple[int, ...] = (1, 3, 5, 10)
MRR_K = 10
METRIC_NAMES: tuple[str, ...] = (
    *(f"hit@{k}" for k in KS),
    *(f"recall@{k}" for k in KS),
    f"mrr@{MRR_K}",
    *(f"ndcg@{k}" for k in KS),
)

Sources = Sequence[Iterable[str]]


def _check(ranked: Sequence[str], sources: Sources, k: int | None = None) -> list[set[str]]:
    sets = [set(s) for s in sources]
    if not sets:
        raise ValueError("item has no gold sources (unanswerable items are not scored)")
    if any(not s for s in sets):
        raise ValueError("a gold source has no relevant chunk (run scripts/validate_eval.py)")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked list contains duplicate chunk ids")
    if k is not None and k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return sets


def relevant_chunks(sources: Sources) -> set[str]:
    """Union of the sources' chunk ids."""
    return set().union(*(set(s) for s in sources))


def hit_at_k(ranked: Sequence[str], sources: Sources, k: int) -> float:
    rel = relevant_chunks(_check(ranked, sources, k))
    return 1.0 if any(cid in rel for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], sources: Sources, k: int) -> float:
    sets = _check(ranked, sources, k)
    top = set(ranked[:k])
    return sum(1 for s in sets if s & top) / len(sets)


def reciprocal_rank(ranked: Sequence[str], sources: Sources, k: int = MRR_K) -> float:
    rel = relevant_chunks(_check(ranked, sources, k))
    for rank, cid in enumerate(ranked[:k], start=1):
        if cid in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], sources: Sources, k: int) -> float:
    rel = relevant_chunks(_check(ranked, sources, k))
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(ranked[:k], start=1) if cid in rel)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(rel)) + 1))
    return dcg / idcg


def item_metrics(ranked: Sequence[str], sources: Sources, ks: Sequence[int] = KS) -> dict[str, float]:
    """Every metric in :data:`METRIC_NAMES` (for ``ks``) for one item."""
    sets = _check(ranked, sources)
    out: dict[str, float] = {}
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, sets, k)
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, sets, k)
    out[f"mrr@{MRR_K}"] = reciprocal_rank(ranked, sets, MRR_K)
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, sets, k)
    return out


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Macro average (each item weighs the same) of per-item metric dicts."""
    if not rows:
        raise ValueError("no items to average")
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}
