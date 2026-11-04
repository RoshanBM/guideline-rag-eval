"""Bootstrap confidence intervals over eval items.

* :func:`bootstrap_ci`: percentile CI of a mean (e.g. a retriever's recall@5 over items).
* :func:`paired_bootstrap`: CI of the mean difference between two systems scored on the same
  items. Each resample draws items, and both systems are scored on the **same** drawn items,
  so item difficulty (shared by both systems) cancels out; this is the bootstrap of the
  per-item differences ``a - b``.

Defaults: 1,000 resamples, 95 % percentile interval, fixed seed (42), so results are
reproducible and every system evaluated on the same items gets the same resamples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 1000
CONFIDENCE = 0.95
SEED = 42


@dataclass(frozen=True)
class Interval:
    estimate: float  # the statistic on the original sample (mean, or mean difference)
    lo: float
    hi: float

    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def _resample_means(values: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    return values[idx].mean(axis=1)


def bootstrap_ci(values: Sequence[float], n_resamples: int = N_RESAMPLES,
                 confidence: float = CONFIDENCE, seed: int = SEED) -> Interval:
    """Mean of ``values`` with a percentile bootstrap CI (items resampled with replacement)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("need a non-empty 1-D sequence of per-item values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    means = _resample_means(arr, n_resamples, seed)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return Interval(float(arr.mean()), float(lo), float(hi))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = N_RESAMPLES,
                     confidence: float = CONFIDENCE, seed: int = SEED) -> Interval:
    """Mean of ``a - b`` (system A minus system B, per item, same item order) with its CI."""
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"paired samples differ in length: {a_arr.shape} vs {b_arr.shape}")
    return bootstrap_ci(a_arr - b_arr, n_resamples, confidence, seed)
