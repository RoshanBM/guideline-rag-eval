"""Refusal-signal metrics on toy scores."""

from __future__ import annotations

import math

import pytest

from src.evaluate.refusal import auroc, auroc_ci, threshold_at_max_false_refusal


def test_auroc_hand_computed():
    assert auroc([3, 4], [1, 2]) == 1.0  # perfect separation
    assert auroc([1, 2], [3, 4]) == 0.0
    # pairs (pos, neg): (1,2)<, (1,0)>, (3,2)>, (3,0)> -> 3/4
    assert auroc([1, 3], [2, 0]) == pytest.approx(0.75)
    assert auroc([2, 2], [2, 2]) == pytest.approx(0.5)  # all ties
    assert auroc([1], [None]) == 1.0  # no retrieved chunk = lowest score


def test_auroc_ci_is_reproducible_and_brackets_the_estimate():
    pos = [0.9, 0.8, 0.85, 0.7, 0.95, 0.6, 0.75, 0.88]
    neg = [0.5, 0.65, 0.4, 0.72]
    a, b = auroc_ci(pos, neg), auroc_ci(pos, neg)
    assert a == b
    assert a.lo <= a.estimate <= a.hi and 0 <= a.lo and a.hi <= 1
    perfect = auroc_ci([3, 4, 5], [0, 1])
    assert (perfect.estimate, perfect.lo, perfect.hi) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        auroc([], [1])


def test_threshold_keeps_false_refusals_at_most_5_percent():
    pos = list(range(10, 50))  # 40 answerable: 5% -> at most 2 refused
    neg = [5, 11, 12, 30]
    t = threshold_at_max_false_refusal(pos, neg, 0.05)
    assert t.threshold == 12  # refuse < 12: answerable 10, 11 refused
    assert t.n_false_refusals == 2 and t.false_refusal_rate == pytest.approx(0.05)
    assert t.n_refused_unanswerable == 2 and t.refusal_rate == pytest.approx(0.5)  # 5 and 11


def test_threshold_with_47_answerable_items():
    pos = [float(i) for i in range(47)]
    t = threshold_at_max_false_refusal(pos, [0.5, 1.5, 100.0], 0.05)
    assert t.n_false_refusals == 2  # floor(0.05 * 47) = 2 (4.3 %)
    assert t.threshold == 2.0 and t.n_refused_unanswerable == 2


def test_threshold_with_ties_never_exceeds_the_limit():
    t = threshold_at_max_false_refusal([1, 1, 1, 2, 3], [0, 1], 0.2)  # may refuse 1 of 5
    assert t.n_false_refusals == 0 and t.threshold == 1  # refusing below 1 refuses no answerable
    assert t.n_refused_unanswerable == 1
    zero = threshold_at_max_false_refusal([1, 2], [0], 0.0)
    assert zero.threshold == 1 and zero.n_false_refusals == 0
    assert math.isinf(threshold_at_max_false_refusal([1], [0], 1.0).threshold)
