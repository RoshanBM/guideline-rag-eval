"""Hybrid fusion (RRF, weighted min-max) and cross-encoder reranking, with fake components."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieve.base import Retriever
from src.retrieve.hybrid import HybridRetriever, min_max, order, rrf_fuse, weighted_fuse
from src.retrieve.rerank import RERANK_SPECS, RerankRetriever


class Fixed:
    """A retriever returning a fixed ranking whatever the query."""

    def __init__(self, name: str, ranking: list[tuple[str, float]]) -> None:
        self.name = name
        self.ranking = ranking
        self.calls: list[int] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        self.calls.append(k)
        return self.ranking[:k]


SPARSE = [("a", 12.0), ("b", 8.0), ("c", 4.0)]           # BM25-like scores
DENSE = [("c", 0.90), ("d", 0.80), ("a", 0.60)]          # cosine-like scores


def test_rrf_hand_computed():
    fused = rrf_fuse([SPARSE, DENSE], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 62)
    # a and c tie on RRF; b and d tie; ties broken by best component rank, then chunk id
    assert [cid for cid, _ in order(fused, [SPARSE, DENSE])] == ["a", "c", "b", "d"]


def test_min_max():
    assert min_max(SPARSE) == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert min_max([("x", 3.0), ("y", 3.0)]) == {"x": 1.0, "y": 1.0}
    assert min_max([]) == {}


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
def test_weighted_hand_computed(alpha):
    fused = weighted_fuse(SPARSE, DENSE, alpha)
    # dense min-max: c 1.0, d 2/3, a 0.0; sparse: a 1.0, b 0.5, c 0.0
    assert fused["a"] == pytest.approx(alpha * 0.0 + (1 - alpha) * 1.0)
    assert fused["b"] == pytest.approx((1 - alpha) * 0.5)
    assert fused["c"] == pytest.approx(alpha * 1.0)
    assert fused["d"] == pytest.approx(alpha * (2 / 3))


def test_weighted_alpha_moves_towards_dense():
    top = lambda a: order(weighted_fuse(SPARSE, DENSE, a), [SPARSE, DENSE])[0][0]  # noqa: E731
    assert top(0.3) == "a" and top(0.7) == "c"
    with pytest.raises(ValueError):
        weighted_fuse(SPARSE, DENSE, 1.5)


def test_hybrid_retriever_fetches_50_from_each_and_records_components():
    sparse, dense = Fixed("bm25", SPARSE), Fixed("bge-small", DENSE)
    h = HybridRetriever("hybrid-rrf", sparse, dense, "rrf")
    assert isinstance(h, Retriever)
    hits = h.search("q", 2)
    assert [cid for cid, _ in hits] == ["a", "c"]
    assert sparse.calls == [50] and dense.calls == [50]
    assert h.last_components["a"] == {"sparse_rank": 1, "sparse_score": 12.0, "dense_rank": 3, "dense_score": 0.6}
    assert h.last_components["b"] == {"sparse_rank": 2, "sparse_score": 8.0}
    with pytest.raises(ValueError):
        HybridRetriever("x", sparse, dense, "weighted")


def test_rerank_resorts_candidates_and_records_scores():
    base = Fixed("hybrid-rrf", [(c, 1.0 / i) for i, c in enumerate("abcdef", start=1)])
    texts = {c: f"text {c}" for c in "abcdef"}
    preferred = {"text e": 5.0, "text b": 3.0}
    scorer = lambda q, passages: np.array([preferred.get(p, 0.0) for p in passages])  # noqa: E731
    r = RerankRetriever("rerank-fake", base, texts, scorer, n_candidates=5)
    hits = r.search("q", 50)
    assert base.calls == [5]  # only the top 5 candidates are reranked and returned
    assert [cid for cid, _ in hits] == ["e", "b", "a", "c", "d"]  # ties keep first-stage order
    assert hits[0] == ("e", 5.0)
    assert len(r.last_candidates) == 5
    assert r.last_candidates[4] == {"chunk_id": "e", "first_stage_rank": 5, "first_stage_score": 0.2,
                                    "rerank_score": 5.0}
    assert r.search("q", 2) == hits[:2] and r.search("q", 0) == []


def test_rerank_specs():
    assert {s.model for s in RERANK_SPECS.values()} == {
        "cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3", "ncbi/MedCPT-Cross-Encoder"}
