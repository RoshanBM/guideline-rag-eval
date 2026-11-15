"""Hybrid retrieval: fuse a sparse (``bm25``, stemmed) and a dense (``bge-small``) ranking.

Each component returns its top ``DEPTH`` = 50 chunks for the query; the two lists are fused
and the top k of the fused list is returned. Registered variants (``src/retrieve/registry.py``):

* ``hybrid-rrf``: Reciprocal Rank Fusion, ``score(c) = sum over lists of 1 / (60 + rank_list(c))``
  (a chunk absent from a list contributes nothing for it).
* ``hybrid-w03`` / ``hybrid-w05`` / ``hybrid-w07``: weighted score fusion with alpha = 0.3, 0.5,
  0.7 (alpha is the **dense** weight): each list's scores are min-max normalised over that
  list's top 50 (``(s - min) / (max - min)``; all 1.0 if every score is equal), a chunk absent
  from a list gets 0 for it, and ``score = alpha * dense + (1 - alpha) * sparse``.

Choosing alpha on the eval set is tuning on test data: all alphas are reported, but RRF (no
parameter chosen on this data; k = 60 is the standard value) is the main hybrid.

Ties in the fused score are broken by the best rank in either list, then by chunk id, so
results are deterministic. :attr:`HybridRetriever.last_components` keeps, for the last query,
each candidate's component ranks and scores (for analysis).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.retrieve.base import Retriever

DEPTH = 50
RRF_K = 60
ALPHAS: dict[str, float] = {"hybrid-w03": 0.3, "hybrid-w05": 0.5, "hybrid-w07": 0.7}

Ranking = Sequence[tuple[str, float]]


def rrf_fuse(rankings: Sequence[Ranking], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal Rank Fusion scores of every chunk in any ranking (ranks are 1-based)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def min_max(ranking: Ranking) -> dict[str, float]:
    """Scores of ``ranking`` scaled to [0, 1] by its own min and max (all 1.0 if they are equal)."""
    if not ranking:
        return {}
    scores = [s for _, s in ranking]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {cid: 1.0 for cid, _ in ranking}
    return {cid: (s - lo) / (hi - lo) for cid, s in ranking}


def weighted_fuse(sparse: Ranking, dense: Ranking, alpha: float) -> dict[str, float]:
    """``alpha * minmax(dense) + (1 - alpha) * minmax(sparse)``; absent from a list counts as 0."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    s, d = min_max(sparse), min_max(dense)
    return {cid: alpha * d.get(cid, 0.0) + (1 - alpha) * s.get(cid, 0.0) for cid in {*s, *d}}


def order(fused: dict[str, float], rankings: Sequence[Ranking]) -> list[tuple[str, float]]:
    """Fused scores sorted best first; ties broken by best component rank, then chunk id."""
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            best_rank[cid] = min(best_rank.get(cid, rank), rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], best_rank.get(kv[0], 10**9), kv[0]))


class HybridRetriever:
    def __init__(self, name: str, sparse: Retriever, dense: Retriever, method: str,
                 alpha: float | None = None, depth: int = DEPTH, rrf_k: int = RRF_K) -> None:
        if method not in ("rrf", "weighted"):
            raise ValueError(f"unknown fusion method {method!r}")
        if method == "weighted" and alpha is None:
            raise ValueError("weighted fusion needs alpha")
        self.name = name
        self.sparse, self.dense = sparse, dense
        self.method, self.alpha, self.depth, self.rrf_k = method, alpha, depth, rrf_k
        self.last_components: dict[str, dict[str, float | int | None]] = {}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        sparse = self.sparse.search(query, self.depth)
        dense = self.dense.search(query, self.depth)
        if self.method == "rrf":
            fused = rrf_fuse([sparse, dense], self.rrf_k)
        else:
            fused = weighted_fuse(sparse, dense, self.alpha)  # type: ignore[arg-type]
        self.last_components = {}
        for label, ranking in (("sparse", sparse), ("dense", dense)):
            for rank, (cid, score) in enumerate(ranking, start=1):
                entry = self.last_components.setdefault(cid, {})
                entry[f"{label}_rank"], entry[f"{label}_score"] = rank, score
        return order(fused, [sparse, dense])[:k]

    def config(self) -> dict[str, object]:
        return {"sparse": self.sparse.name, "dense": self.dense.name, "method": self.method,
                "alpha_dense": self.alpha, "rrf_k": self.rrf_k if self.method == "rrf" else None,
                "component_depth": self.depth}
