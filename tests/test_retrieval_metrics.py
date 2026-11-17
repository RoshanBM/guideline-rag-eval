"""Retrieval metrics against hand-computed examples (binary relevance, 1-based ranks).

Gold sources are given as one chunk-id set per source. Recall@k is per gold source (a source is
found if any chunk holding its span is in the top k); hit@k, MRR@10 and nDCG@k use the union of
relevant chunks.

Discount used by nDCG: 1/log2(r + 1), so rank 1 -> 1, rank 2 -> 1/log2 3 = 0.630930,
rank 4 -> 1/log2 5 = 0.430677, rank 10 -> 1/log2 11 = 0.289065.
"""

from __future__ import annotations

import pytest

from src.evaluate.retrieval_metrics import (
    METRIC_NAMES,
    hit_at_k,
    item_metrics,
    mean_metrics,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

FILLER = [f"x{i}" for i in range(1, 60)]  # irrelevant chunk ids


def ranking(**at_rank: int) -> list[str]:
    """A ranked list of length 50 with the named chunk ids at the given 1-based ranks."""
    ranked = FILLER[:50]
    for cid, rank in at_rank.items():
        ranked[rank - 1] = cid
    return ranked


def test_single_relevant_at_rank_1():
    m = item_metrics(ranking(g=1), [{"g"}])
    assert all(v == pytest.approx(1.0) for v in m.values())
    assert list(m) == list(METRIC_NAMES)


def test_single_relevant_at_rank_10():
    m = item_metrics(ranking(g=10), [{"g"}])
    for k in (1, 3, 5):
        assert m[f"hit@{k}"] == 0 and m[f"recall@{k}"] == 0 and m[f"ndcg@{k}"] == 0
    assert m["hit@10"] == 1 and m["recall@10"] == 1
    assert m["mrr@10"] == pytest.approx(0.1)
    assert m["ndcg@10"] == pytest.approx(0.289065, abs=1e-6)  # (1/log2 11) / 1


def test_relevant_just_outside_top_10_scores_zero():
    m = item_metrics(ranking(g=11), [{"g"}])
    assert all(v == 0 for v in m.values())


def test_miss_no_relevant_chunk_retrieved():
    m = item_metrics(["a", "b", "c"], [{"g"}])
    assert all(v == 0 for v in m.values())


def test_two_sources_in_two_chunks_at_ranks_2_and_4():
    ranked, sources = ranking(g1=2, g2=4), [{"g1"}, {"g2"}]
    assert [hit_at_k(ranked, sources, k) for k in (1, 3, 5, 10)] == [0, 1, 1, 1]
    assert [recall_at_k(ranked, sources, k) for k in (1, 3, 5, 10)] == [0, 0.5, 1, 1]
    assert reciprocal_rank(ranked, sources) == pytest.approx(0.5)
    assert ndcg_at_k(ranked, sources, 1) == 0
    # DCG@3 = 1/log2 3; IDCG@3 = 1 + 1/log2 3 (two relevant chunks)
    assert ndcg_at_k(ranked, sources, 3) == pytest.approx(0.630930 / 1.630930, abs=1e-6)  # 0.386853
    # DCG@5 = 1/log2 3 + 1/log2 5
    assert ndcg_at_k(ranked, sources, 5) == pytest.approx(1.061606 / 1.630930, abs=1e-6)  # 0.650921
    assert ndcg_at_k(ranked, sources, 10) == pytest.approx(ndcg_at_k(ranked, sources, 5))


def test_combined_question_only_one_source_retrieved():
    """A combined question needs both sources: one found at rank 1 gives half recall."""
    ranked, sources = ranking(g1=1), [{"g1"}, {"g2"}]
    m = item_metrics(ranked, sources)
    assert [m[f"hit@{k}"] for k in (1, 3, 5, 10)] == [1, 1, 1, 1]
    assert [m[f"recall@{k}"] for k in (1, 3, 5, 10)] == [0.5, 0.5, 0.5, 0.5]
    assert m["mrr@10"] == 1
    assert m["ndcg@1"] == pytest.approx(1.0)  # IDCG@1 = 1: only one slot
    assert m["ndcg@3"] == pytest.approx(1 / 1.630930, abs=1e-6)  # 0.613147


def test_two_sources_found_at_ranks_1_and_2_are_perfect():
    m = item_metrics(ranking(g1=1, g2=2), [{"g1"}, {"g2"}])
    assert m["recall@1"] == 0.5 and m["recall@3"] == 1
    assert all(m[f"ndcg@{k}"] == pytest.approx(1.0) for k in (1, 3, 5, 10))


# aw-008 shape: source 1 is only in the adult table (A); sources 2 and 3 are footnotes printed in
# both the adult (A) and the child (C) table, so either chunk holds them.
AW008 = [{"A"}, {"A", "C"}, {"A", "C"}]


def test_span_shared_by_two_chunks_is_found_by_either():
    m = item_metrics(ranking(A=1), AW008)
    assert all(m[f"recall@{k}"] == 1 for k in (1, 3, 5, 10))  # child chunk not needed for recall
    assert m["hit@1"] == 1 and m["mrr@10"] == 1
    # nDCG still counts relevant chunks: A and C are both relevant, only A retrieved
    assert m["ndcg@1"] == pytest.approx(1.0)
    assert m["ndcg@10"] == pytest.approx(1 / 1.630930, abs=1e-6)


def test_span_shared_by_two_chunks_only_other_chunk_retrieved():
    m = item_metrics(ranking(C=2), AW008)
    assert m["recall@1"] == 0
    assert m["recall@3"] == pytest.approx(2 / 3)  # sources 2 and 3 found via C, source 1 not
    assert m["hit@3"] == 1 and m["mrr@10"] == pytest.approx(0.5)
    assert m["ndcg@3"] == pytest.approx(0.630930 / 1.630930, abs=1e-6)


def test_two_sources_in_the_same_chunk():
    m = item_metrics(ranking(T=3), [{"T"}, {"T"}])
    assert m["recall@1"] == 0 and m["recall@3"] == 1
    assert m["ndcg@3"] == pytest.approx(0.5)  # (1/log2 4) / 1


def test_short_ranked_list():
    assert recall_at_k(["a", "g"], [{"g"}], 10) == 1
    assert ndcg_at_k(["a", "g"], [{"g"}], 10) == pytest.approx(0.630930, abs=1e-6)


def test_unanswerable_items_are_rejected():
    with pytest.raises(ValueError, match="no gold sources"):
        item_metrics(["a", "b"], [])
    with pytest.raises(ValueError, match="no relevant chunk"):
        item_metrics(["a", "b"], [{"a"}, set()])


def test_duplicates_and_bad_k_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        recall_at_k(["a", "a"], [{"a"}], 2)
    with pytest.raises(ValueError, match="k must be"):
        hit_at_k(["a"], [{"a"}], 0)


def test_mean_metrics_is_a_macro_average():
    rows = [item_metrics(ranking(g=1), [{"g"}]), item_metrics(["a"], [{"g"}])]
    mean = mean_metrics(rows)
    assert mean["recall@10"] == 0.5 and mean["mrr@10"] == 0.5
    with pytest.raises(ValueError):
        mean_metrics([])
