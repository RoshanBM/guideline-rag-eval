"""BM25 retrieval over the chunk corpus, with ``bm25s``.

Two variants are registered (``src/retrieve/registry.py``):

* ``bm25``: Snowball-stemmed tokens (``tokenize(..., stem=True)``), the baseline;
* ``bm25-nostem``: the same without stemming (ablation).

The index is built from :func:`src.retrieve.corpus.load_corpus` (retrieval ``text`` of the
``index_default`` chunks) and saved to ``data/index/<name>/`` (git-ignored) together with
``meta.json``: the chunk ids in index order, the BM25 parameters, the stemming flag, the
tokenizer version and a corpus fingerprint. :meth:`BM25Retriever.load` refuses an index whose
fingerprint, tokenizer version or stemming flag does not match.

Ranking: every chunk is scored and the scores are sorted stably, so ties keep corpus order and
runs are deterministic. Chunks scoring 0 (no query term in common) are not returned.
"""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import numpy as np

from src.parse.sources import PROJECT_ROOT
from src.retrieve.corpus import CorpusChunk, corpus_fingerprint
from src.retrieve.tokenize import TOKENIZER_VERSION, tokenize

INDEX_ROOT: Path = PROJECT_ROOT / "data" / "index"
META_NAME = "meta.json"
DEFAULT_PARAMS: dict[str, object] = {"method": "lucene", "k1": 1.5, "b": 0.75}


def variant_name(stem: bool) -> str:
    return "bm25" if stem else "bm25-nostem"


def index_dir_for(stem: bool) -> Path:
    return INDEX_ROOT / variant_name(stem)


class StaleIndexError(RuntimeError):
    """A saved index does not match the current corpus or tokenizer."""


class BM25Retriever:
    def __init__(self, model: bm25s.BM25, chunk_ids: list[str], params: dict[str, object],
                 fingerprint: str, stem: bool) -> None:
        self._model = model
        self.chunk_ids = list(chunk_ids)
        self.params = dict(params)
        self.fingerprint = fingerprint
        self.stem = stem
        self.name = variant_name(stem)

    @classmethod
    def build(cls, corpus: list[CorpusChunk], stem: bool = True, **params: object) -> BM25Retriever:
        """Index ``corpus`` (``params`` override :data:`DEFAULT_PARAMS`)."""
        merged = {**DEFAULT_PARAMS, **params}
        model = bm25s.BM25(method=merged["method"], k1=merged["k1"], b=merged["b"])
        model.index([tokenize(chunk.text, stem=stem) for chunk in corpus], show_progress=False)
        return cls(model, [c.chunk_id for c in corpus], merged, corpus_fingerprint(corpus), stem)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        tokens = tokenize(query, stem=self.stem)
        if not tokens:
            return []
        scores = np.asarray(self._model.get_scores(tokens), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, index_dir: Path | None = None) -> None:
        index_dir = index_dir or index_dir_for(self.stem)
        index_dir.mkdir(parents=True, exist_ok=True)
        self._model.save(str(index_dir), show_progress=False)
        meta = {
            "name": self.name,
            "chunk_ids": self.chunk_ids,
            "params": self.params,
            "stem": self.stem,
            "tokenizer_version": TOKENIZER_VERSION,
            "corpus_fingerprint": self.fingerprint,
            "bm25s_version": bm25s.__version__,
        }
        (index_dir / META_NAME).write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path | None = None, corpus: list[CorpusChunk] | None = None,
             stem: bool = True) -> BM25Retriever:
        """Load a saved index; raise :class:`StaleIndexError` unless it matches ``stem`` (and ``corpus``)."""
        index_dir = index_dir or index_dir_for(stem)
        meta_path = index_dir / META_NAME
        if not meta_path.exists():
            raise FileNotFoundError(f"no BM25 index in {index_dir}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("tokenizer_version") != TOKENIZER_VERSION:
            raise StaleIndexError(
                f"index built with tokenizer v{meta.get('tokenizer_version')}, current is v{TOKENIZER_VERSION}")
        if meta.get("stem") != stem:
            raise StaleIndexError(f"index in {index_dir} has stem={meta.get('stem')}, wanted stem={stem}")
        if corpus is not None and meta["corpus_fingerprint"] != corpus_fingerprint(corpus):
            raise StaleIndexError(f"index in {index_dir} was built from different chunks; rebuild it")
        model = bm25s.BM25.load(str(index_dir), show_progress=False)
        return cls(model, meta["chunk_ids"], meta["params"], meta["corpus_fingerprint"], stem)

    @classmethod
    def load_or_build(cls, corpus: list[CorpusChunk], stem: bool = True,
                      index_dir: Path | None = None) -> BM25Retriever:
        """The saved index if it matches ``corpus``, else a fresh one (which is then saved)."""
        try:
            return cls.load(index_dir, corpus, stem)
        except (FileNotFoundError, StaleIndexError):
            retriever = cls.build(corpus, stem=stem)
            retriever.save(index_dir)
            return retriever
