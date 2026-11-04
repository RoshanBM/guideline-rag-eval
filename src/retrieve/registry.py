"""Named retrievers for the evaluation harness: ``build_retriever(name, corpus)``.

``bm25`` (stemmed, the baseline) and ``bm25-nostem`` (ablation), and the dense models of
``src/retrieve/dense.py``. Dense imports (torch, FAISS) are deferred so BM25 runs without them.

Building always starts from scratch so the reported build / encode time is a fresh measurement;
the index or embeddings are then saved to ``data/index/<name>/``. For BM25 the saved index is
loaded back and that copy is evaluated, which also checks the save/load path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.retrieve.base import Retriever
from src.retrieve.corpus import CorpusChunk

BM25_NAMES: tuple[str, ...] = ("bm25", "bm25-nostem")
DENSE_NAMES: tuple[str, ...] = ("bge-small", "bge-base", "pubmedbert", "medcpt")
RETRIEVER_NAMES: tuple[str, ...] = (*BM25_NAMES, *DENSE_NAMES)


@dataclass
class BuildResult:
    retriever: Retriever
    build_s: float  # BM25: index build; dense: corpus encode (model load excluded)
    device: str
    config: dict[str, Any] = field(default_factory=dict)


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
        from dataclasses import asdict

        from src.retrieve.dense import DENSE_SPECS, DenseRetriever, device_label

        spec = DENSE_SPECS[name]
        retriever = DenseRetriever.build(spec, corpus)
        retriever.save()
        return BuildResult(retriever, retriever.timings["corpus_encode_s"], device_label(retriever.device), {
            "spec": asdict(spec),
            "model_load_s": round(retriever.timings["model_load_s"], 3),
            "similarity": "inner product on L2-normalised embeddings (faiss.IndexFlatIP)",
            "dim": int(retriever.embeddings.shape[1]),
        })
    raise ValueError(f"unknown retriever {name!r}; choose from {RETRIEVER_NAMES}")
