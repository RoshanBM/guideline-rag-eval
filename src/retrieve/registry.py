"""Named retrievers for the evaluation harness: ``build_retriever(name, corpus)``.

* BM25: ``bm25`` (stemmed, the baseline), ``bm25-nostem`` (ablation);
* dense (``src/retrieve/dense.py``): ``bge-small``, ``bge-base``, ``pubmedbert``, ``medcpt``,
  ``medcpt-raw``;
* hybrid (``src/retrieve/hybrid.py``, bm25 + bge-small): ``hybrid-rrf`` (main),
  ``hybrid-w03``, ``hybrid-w05``, ``hybrid-w07``;
* rerankers (``src/retrieve/rerank.py``, top 30 of hybrid-rrf): ``rerank-minilm``,
  ``rerank-bge``, ``rerank-medcpt``.

Heavy imports (torch, FAISS, models) are deferred so BM25 runs without them.

A standalone retriever is built from scratch so its build / encode time is a fresh measurement;
its index or embeddings are then saved to ``data/index/<name>/``. For BM25 the saved index is
loaded back and that copy is evaluated, which also checks the save/load path. Hybrids and
rerankers use **shared** components (``bm25``, ``bge-small``, ``hybrid-rrf``), built once per
process; their ``build_s`` is the sum of the components' build times. Call
:func:`release_shared` when done to free GPU memory.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.retrieve.base import Retriever
from src.retrieve.corpus import CorpusChunk, corpus_fingerprint

BM25_NAMES: tuple[str, ...] = ("bm25", "bm25-nostem")
DENSE_NAMES: tuple[str, ...] = ("bge-small", "bge-base", "pubmedbert", "medcpt", "medcpt-raw")
HYBRID_NAMES: tuple[str, ...] = ("hybrid-rrf", "hybrid-w03", "hybrid-w05", "hybrid-w07")
RERANK_NAMES: tuple[str, ...] = ("rerank-minilm", "rerank-bge", "rerank-medcpt")
RETRIEVER_NAMES: tuple[str, ...] = (*BM25_NAMES, *DENSE_NAMES, *HYBRID_NAMES, *RERANK_NAMES)

HYBRID_SPARSE, HYBRID_DENSE, RERANK_BASE = "bm25", "bge-small", "hybrid-rrf"


@dataclass
class BuildResult:
    retriever: Retriever
    build_s: float  # BM25: index build; dense: corpus encode (model load excluded); hybrid/rerank: components
    device: str
    config: dict[str, Any] = field(default_factory=dict)
    shared: bool = False  # owned by the shared-component cache: do not close after one run


_SHARED: dict[tuple[str, str], BuildResult] = {}


def _shared(name: str, corpus: list[CorpusChunk]) -> BuildResult:
    key = (name, corpus_fingerprint(corpus))
    if key not in _SHARED:
        built = build_retriever(name, corpus)
        built.shared = True
        _SHARED[key] = built
    return _SHARED[key]


def release_shared() -> None:
    """Close and forget every shared component (frees GPU memory)."""
    for built in _SHARED.values():
        close = getattr(built.retriever, "close", None)
        if close is not None:
            close()
    _SHARED.clear()


def build_retriever(name: str, corpus: list[CorpusChunk]) -> BuildResult:
    if name in BM25_NAMES:
        from src.retrieve.bm25 import BM25Retriever, index_dir_for

        stem = name == "bm25"
        t0 = time.perf_counter()
        built = BM25Retriever.build(corpus, stem=stem)
        build_s = time.perf_counter() - t0
        built.save()
        loaded = BM25Retriever.load(corpus=corpus, stem=stem)
        return BuildResult(loaded, build_s, "cpu", {
            "params": loaded.params, "stem": stem, "index_dir": index_dir_for(stem).as_posix(),
        })

    if name in DENSE_NAMES:
        from src.retrieve.dense import DENSE_SPECS, DenseRetriever, device_label

        spec = DENSE_SPECS[name]
        retriever = DenseRetriever.build(spec, corpus)
        retriever.save()
        return BuildResult(retriever, retriever.timings["corpus_encode_s"], device_label(retriever.device), {
            "spec": asdict(spec),
            "model_load_s": round(retriever.timings["model_load_s"], 3),
            "similarity": ("inner product on L2-normalised embeddings" if spec.normalise
                           else "raw (unnormalised) inner product") + " (faiss.IndexFlatIP)",
            "dim": int(retriever.embeddings.shape[1]),
        })

    if name in HYBRID_NAMES:
        from src.retrieve.hybrid import ALPHAS, HybridRetriever

        sparse, dense = _shared(HYBRID_SPARSE, corpus), _shared(HYBRID_DENSE, corpus)
        if name == "hybrid-rrf":
            retriever = HybridRetriever(name, sparse.retriever, dense.retriever, "rrf")
        else:
            retriever = HybridRetriever(name, sparse.retriever, dense.retriever, "weighted", ALPHAS[name])
        return BuildResult(retriever, sparse.build_s + dense.build_s, dense.device, {
            **retriever.config(),
            "component_build_s": {HYBRID_SPARSE: round(sparse.build_s, 4), HYBRID_DENSE: round(dense.build_s, 4)},
        })

    if name in RERANK_NAMES:
        from src.retrieve.dense import default_device, device_label
        from src.retrieve.rerank import N_CANDIDATES, RERANK_SPECS, CrossEncoderScorer, RerankRetriever

        base = _shared(RERANK_BASE, corpus)
        spec = RERANK_SPECS[name]
        device = default_device()
        t0 = time.perf_counter()
        scorer = CrossEncoderScorer(spec, device)
        model_load_s = time.perf_counter() - t0
        texts = {c.chunk_id: c.text for c in corpus}
        retriever = RerankRetriever(name, base.retriever, texts, scorer, N_CANDIDATES)
        return BuildResult(retriever, base.build_s, device_label(device), {
            "first_stage": RERANK_BASE,
            "n_candidates": N_CANDIDATES,
            "spec": asdict(spec),
            "scores": "raw logits (no sigmoid)",
            "model_load_s": round(model_load_s, 3),
        })

    raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVER_NAMES}")
