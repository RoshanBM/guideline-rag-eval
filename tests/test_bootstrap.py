"""Bootstrap CIs on toy data."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate.bootstrap import bootstrap_ci, paired_bootstrap


def test_constant_values_give_a_degenerate_interval():
    ci = bootstrap_ci([0.7] * 20)
    assert (ci.estimate, ci.lo, ci.hi) == pytest.approx((0.7, 0.7, 0.7))


def test_interval_contains_the_mean_and_is_reproducible():
    values = [1, 0] * 25  # mean 0.5, n = 50
    a, b = bootstrap_ci(values), bootstrap_ci(values)
    assert a == b  # fixed seed
    assert a.estimate == 0.5
    assert 0 <= a.lo < 0.5 < a.hi <= 1
    # normal approximation: SE = sqrt(0.25 / 50) = 0.0707, so the CI is about 0.5 +- 0.139
    assert a.lo == pytest.approx(0.36, abs=0.03) and a.hi == pytest.approx(0.64, abs=0.03)
    continuous = np.random.default_rng(1).uniform(size=50)
    assert bootstrap_ci(continuous, seed=7) != bootstrap_ci(continuous)  # the seed matters


def test_paired_difference_of_shifted_systems_is_exact():
    """B = A - 0.1 on every item: the paired difference has no spread at all."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, 1.0, size=30)
    ci = paired_bootstrap(a, a - 0.1)
    assert (ci.estimate, ci.lo, ci.hi) == pytest.approx((0.1, 0.1, 0.1))
    assert ci.excludes_zero()


def test_pairing_removes_item_difficulty():
    """Item difficulty varies a lot but A beats B by 0.05 on every item but one: the paired CI
    is narrow and excludes 0, while the two unpaired CIs overlap."""
    difficulty = np.linspace(0, 0.9, 40)
    a = difficulty + 0.05
    b = difficulty.copy()
    b[0] += 0.1  # one item where B is better
    paired = paired_bootstrap(a, b)
    assert paired.estimate == pytest.approx((0.05 * 39 - 0.05) / 40)
    assert paired.excludes_zero()
    ca, cb = bootstrap_ci(a), bootstrap_ci(b)
    assert ca.lo < cb.hi  # unpaired intervals overlap


def test_no_difference_gives_an_interval_around_zero():
    a = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    b = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0]  # same mean, different items
    ci = paired_bootstrap(a, b)
    assert ci.estimate == 0 and ci.lo < 0 < ci.hi and not ci.excludes_zero()


def test_bad_input():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])
    with pytest.raises(ValueError):
        bootstrap_ci([1, 2], confidence=1.5)
