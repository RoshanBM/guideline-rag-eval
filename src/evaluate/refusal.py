"""Top-1 retrieval score as a refusal signal: does it separate answerable from unanswerable items?

* :func:`auroc`: probability that a random answerable item has a higher top-1 score than a
  random unanswerable one (ties count one half), i.e. the Mann-Whitney AUC. 0.5 = no signal.
* :func:`auroc_ci`: percentile bootstrap CI, resampling answerable and unanswerable items
  separately (stratified, so every resample keeps 47 / 15 items), 1,000 resamples, seed 42.
* :func:`threshold_at_max_false_refusal`: the threshold that refuses as many unanswerable items
  as possible while refusing at most ``max_false_refusal`` of the answerable ones; an item is
  refused when its top-1 score is **below** the threshold. Chosen on the eval set itself, so the
  rates are optimistic for unseen questions.

An item with no retrieved chunk (``None`` score, e.g. BM25 with no query term in the corpus) is
treated as the lowest possible score.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.evaluate.bootstrap import CONFIDENCE, N_RESAMPLES, SEED, Interval


def _clean(scores: Sequence[float | None]) -> np.ndarray:
    return np.array([-math.inf if s is None else float(s) for s in scores], dtype=np.float64)


def _auroc_arrays(pos: np.ndarray, neg: np.ndarray) -> float:
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def auroc(pos_scores: Sequence[float | None], neg_scores: Sequence[float | None]) -> float:
    """AUROC with answerable items (``pos``) expected to score higher than unanswerable (``neg``)."""
    pos, neg = _clean(pos_scores), _clean(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need at least one answerable and one unanswerable score")
    return _auroc_arrays(pos, neg)


def auroc_ci(pos_scores: Sequence[float | None], neg_scores: Sequence[float | None],
             n_resamples: int = N_RESAMPLES, confidence: float = CONFIDENCE, seed: int = SEED) -> Interval:
    pos, neg = _clean(pos_scores), _clean(neg_scores)
    estimate = auroc(pos, neg)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        p = pos[rng.integers(0, len(pos), len(pos))]
        n = neg[rng.integers(0, len(neg), len(neg))]
        stats[i] = _auroc_arrays(p, n)
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return Interval(estimate, float(lo), float(hi))


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float  # refuse when top-1 score < threshold
    false_refusal_rate: float  # answerable items refused
    refusal_rate: float  # unanswerable items refused
    n_false_refusals: int
    n_refused_unanswerable: int


def threshold_at_max_false_refusal(pos_scores: Sequence[float | None], neg_scores: Sequence[float | None],
                                   max_false_refusal: float = 0.05) -> ThresholdResult:
    """Highest threshold whose false-refusal rate on the answerable items is <= ``max_false_refusal``."""
    pos, neg = np.sort(_clean(pos_scores)), _clean(neg_scores)
    allowed = math.floor(max_false_refusal * len(pos) + 1e-9)  # answerable items we may refuse
    # refusing everything strictly below pos[allowed] refuses at most `allowed` answerable items
    threshold = float(pos[allowed]) if allowed < len(pos) else math.inf
    n_false = int((pos < threshold).sum())
    n_refused = int((neg < threshold).sum())
    return ThresholdResult(threshold, n_false / len(pos), n_refused / len(neg), n_false, n_refused)
