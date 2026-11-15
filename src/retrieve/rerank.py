"""Cross-encoder reranking of the top ``N_CANDIDATES`` = 30 chunks from ``hybrid-rrf``.

Registered rerankers (``src/retrieve/registry.py``):

* ``rerank-minilm``: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
* ``rerank-bge``: ``BAAI/bge-reranker-v2-m3``
* ``rerank-medcpt``: ``ncbi/MedCPT-Cross-Encoder``

Each (query, chunk ``text``) pair is scored by the cross-encoder (sentence-transformers
``CrossEncoder``, max 512 tokens, the chunk side truncated first) and the candidates are
re-sorted by that score; ties keep the first-stage order. Scores are the model's **raw logits**
(no sigmoid), so they are comparable within a model, not across models.

Only the 30 candidates are returned, so a gold chunk the first stage ranks below 30 is
"not retrieved" for a reranker. :attr:`RerankRetriever.last_candidates` keeps, for the last
query, every candidate with its first-stage rank and score and its reranker score.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from src.retrieve.base import Retriever

N_CANDIDATES = 30
MAX_LENGTH = 512


@dataclass(frozen=True)
class RerankSpec:
    name: str
    model: str
    max_length: int = MAX_LENGTH


RERANK_SPECS: dict[str, RerankSpec] = {
    "rerank-minilm": RerankSpec("rerank-minilm", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    "rerank-bge": RerankSpec("rerank-bge", "BAAI/bge-reranker-v2-m3"),
    "rerank-medcpt": RerankSpec("rerank-medcpt", "ncbi/MedCPT-Cross-Encoder"),
}

Scorer = Callable[[str, Sequence[str]], np.ndarray]


class CrossEncoderScorer:
    """Raw-logit scores of (query, passage) pairs with a sentence-transformers CrossEncoder."""

    def __init__(self, spec: RerankSpec, device: str, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(spec.model, device=device, max_length=spec.max_length,
                                  activation_fn=torch.nn.Identity())
        if self.model.num_labels != 1:
            raise ValueError(f"{spec.model}: expected a single-logit cross-encoder, got {self.model.num_labels}")
        self.batch_size = batch_size

    def __call__(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size,
                                    show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(scores, dtype=np.float64).reshape(-1)


class RerankRetriever:
    def __init__(self, name: str, base: Retriever, texts: dict[str, str], scorer: Scorer,
                 n_candidates: int = N_CANDIDATES) -> None:
        self.name = name
        self.base = base
        self.texts = texts
        self.scorer = scorer
        self.n_candidates = n_candidates
        self.last_candidates: list[dict[str, object]] = []

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        candidates = self.base.search(query, self.n_candidates)
        scores = self.scorer(query, [self.texts[cid] for cid, _ in candidates])
        self.last_candidates = [
            {"chunk_id": cid, "first_stage_rank": rank, "first_stage_score": round(float(s0), 6),
             "rerank_score": round(float(s), 6)}
            for rank, ((cid, s0), s) in enumerate(zip(candidates, scores), start=1)
        ]
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        return [(candidates[i][0], float(scores[i])) for i in order[:k]]

    def close(self) -> None:
        self.scorer = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
