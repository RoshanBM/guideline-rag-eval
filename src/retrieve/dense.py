"""Dense retrieval: embed each chunk's retrieval ``text`` and search with FAISS.

Embeddings are L2-normalised and searched by inner product (``faiss.IndexFlatIP``), i.e. cosine
similarity, for every model. Registered models (:data:`DENSE_SPECS`):

* ``bge-small`` / ``bge-base``: ``BAAI/bge-{small,base}-en-v1.5`` via sentence-transformers.
  Queries get the model card's retrieval instruction ("Represent this sentence for searching
  relevant passages: "); chunks get no instruction.
* ``pubmedbert``: ``NeuML/pubmedbert-base-embeddings`` via sentence-transformers (mean
  pooling), no prefix.
* ``medcpt``: ``ncbi/MedCPT-Query-Encoder`` for queries (max 64 tokens) and
  ``ncbi/MedCPT-Article-Encoder`` for chunks (max 512 tokens), via ``transformers``, [CLS]
  embedding, as on the model cards. A chunk is encoded as the pair
  ``[context_header, rest of text]``, as the card does for ``[title, abstract]``; chunks
  without a header (all hypertension chunks) are encoded as ``["", text]`` so the body stays in
  the second segment. Note: MedCPT was trained for raw inner product; normalising (as for the
  other models) can change its ranking slightly.

Embeddings are cached in ``data/index/<name>/`` (``embeddings.npy`` + ``meta.json`` with the
model ids, prefix, corpus fingerprint, device and encode time). :meth:`DenseRetriever.load_or_build`
reuses the cache only when the spec and the corpus fingerprint match.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers.utils import logging as hf_logging

from src.parse.sources import PROJECT_ROOT
from src.retrieve.corpus import CorpusChunk, corpus_fingerprint

hf_logging.set_verbosity_error()  # silences "sequence longer than max length" when counting tokens

INDEX_ROOT: Path = PROJECT_ROOT / "data" / "index"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class DenseSpec:
    name: str
    kind: str  # "st" (sentence-transformers, one model) or "medcpt" (two encoders, transformers)
    doc_model: str
    query_model: str
    query_prefix: str = ""
    query_max_length: int | None = None  # None: the model's own max_seq_length
    doc_max_length: int | None = None


DENSE_SPECS: dict[str, DenseSpec] = {
    "bge-small": DenseSpec("bge-small", "st", "BAAI/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5",
                           query_prefix=BGE_QUERY_PREFIX),
    "bge-base": DenseSpec("bge-base", "st", "BAAI/bge-base-en-v1.5", "BAAI/bge-base-en-v1.5",
                          query_prefix=BGE_QUERY_PREFIX),
    "pubmedbert": DenseSpec("pubmedbert", "st", "NeuML/pubmedbert-base-embeddings",
                            "NeuML/pubmedbert-base-embeddings"),
    "medcpt": DenseSpec("medcpt", "medcpt", "ncbi/MedCPT-Article-Encoder", "ncbi/MedCPT-Query-Encoder",
                        query_max_length=64, doc_max_length=512),
}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_label(device: str) -> str:
    if device.startswith("cuda"):
        return f"cuda ({torch.cuda.get_device_name(torch.device(device))})"
    return "cpu"


def l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.ascontiguousarray(x / np.maximum(norms, 1e-12), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #
class _STEncoder:
    """One sentence-transformers model for queries and chunks."""

    def __init__(self, spec: DenseSpec, device: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.model = SentenceTransformer(spec.doc_model, device=device)
        self.doc_max_length = spec.doc_max_length or self.model.max_seq_length
        self.query_max_length = spec.query_max_length or self.model.max_seq_length
        self.model.max_seq_length = self.doc_max_length

    def encode_docs(self, chunks: list[CorpusChunk], batch_size: int) -> np.ndarray:
        return self.model.encode([c.text for c in chunks], batch_size=batch_size,
                                 convert_to_numpy=True, show_progress_bar=False)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode([self.spec.query_prefix + q for q in queries],
                                 convert_to_numpy=True, show_progress_bar=False)

    def doc_token_counts(self, chunks: list[CorpusChunk]) -> list[int]:
        tok = self.model.tokenizer
        return [len(tok(c.text, add_special_tokens=True)["input_ids"]) for c in chunks]

    def query_token_counts(self, queries: list[str]) -> list[int]:
        tok = self.model.tokenizer
        return [len(tok(self.spec.query_prefix + q, add_special_tokens=True)["input_ids"]) for q in queries]


class _MedCPTEncoder:
    """MedCPT: separate query and article encoders, [CLS] pooling (transformers)."""

    def __init__(self, spec: DenseSpec, device: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.device = device
        self.q_tok = AutoTokenizer.from_pretrained(spec.query_model)
        self.q_model = AutoModel.from_pretrained(spec.query_model).to(device).eval()
        self.d_tok = AutoTokenizer.from_pretrained(spec.doc_model)
        self.d_model = AutoModel.from_pretrained(spec.doc_model).to(device).eval()
        self.doc_max_length = spec.doc_max_length or 512
        self.query_max_length = spec.query_max_length or 64

    @torch.no_grad()
    def encode_docs(self, chunks: list[CorpusChunk], batch_size: int) -> np.ndarray:
        out = []
        for i in range(0, len(chunks), batch_size):
            pairs = [c.header_and_body() for c in chunks[i:i + batch_size]]
            enc = self.d_tok([h for h, _ in pairs], [b for _, b in pairs], truncation=True, padding=True,
                             return_tensors="pt", max_length=self.doc_max_length).to(self.device)
            out.append(self.d_model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        enc = self.q_tok(queries, truncation=True, padding=True, return_tensors="pt",
                         max_length=self.query_max_length).to(self.device)
        return self.q_model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()

    def doc_token_counts(self, chunks: list[CorpusChunk]) -> list[int]:
        return [len(self.d_tok(*c.header_and_body())["input_ids"]) for c in chunks]

    def query_token_counts(self, queries: list[str]) -> list[int]:
        return [len(self.q_tok(q)["input_ids"]) for q in queries]


def make_encoder(spec: DenseSpec, device: str) -> _STEncoder | _MedCPTEncoder:
    if spec.kind == "st":
        return _STEncoder(spec, device)
    if spec.kind == "medcpt":
        return _MedCPTEncoder(spec, device)
    raise ValueError(f"unknown encoder kind {spec.kind!r}")


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class DenseRetriever:
    def __init__(self, spec: DenseSpec, encoder: _STEncoder | _MedCPTEncoder, embeddings: np.ndarray,
                 chunk_ids: list[str], fingerprint: str, device: str) -> None:
        if len(chunk_ids) != len(embeddings):
            raise ValueError("one embedding per chunk id expected")
        self.spec = spec
        self.name = spec.name
        self.encoder = encoder
        self.embeddings = l2_normalise(embeddings)
        self.chunk_ids = list(chunk_ids)
        self.fingerprint = fingerprint
        self.device = device
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.timings: dict[str, float] = {}

    @classmethod
    def build(cls, spec: DenseSpec, corpus: list[CorpusChunk], device: str | None = None,
              batch_size: int = 32) -> DenseRetriever:
        """Load the model(s) and encode ``corpus``. ``timings``: model load and corpus encode (s)."""
        device = device or default_device()
        t0 = time.perf_counter()
        encoder = make_encoder(spec, device)
        model_load_s = time.perf_counter() - t0
        encoder.encode_docs(corpus[:2], batch_size)  # warm-up (CUDA init), not timed
        encoder.encode_queries(["warm-up"])
        _sync(device)
        t0 = time.perf_counter()
        emb = encoder.encode_docs(corpus, batch_size)
        _sync(device)
        encode_s = time.perf_counter() - t0
        retriever = cls(spec, encoder, emb, [c.chunk_id for c in corpus], corpus_fingerprint(corpus), device)
        retriever.timings = {"model_load_s": model_load_s, "corpus_encode_s": encode_s}
        return retriever

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        q = l2_normalise(self.encoder.encode_queries([query]))
        scores, idx = self.index.search(q, min(k, self.index.ntotal))
        return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def truncation_report(self, corpus: list[CorpusChunk], queries: list[str]) -> dict[str, Any]:
        """How many chunks / queries exceed the encoder's max sequence length (tokens incl. specials)."""
        doc_counts = self.encoder.doc_token_counts(corpus)
        q_counts = self.encoder.query_token_counts(queries)
        over = [(c.chunk_id, n) for c, n in zip(corpus, doc_counts) if n > self.encoder.doc_max_length]
        return {
            "doc_max_length": self.encoder.doc_max_length,
            "n_chunks": len(corpus),
            "n_chunks_truncated": len(over),
            "max_chunk_tokens": max(doc_counts),
            "median_chunk_tokens": float(np.median(doc_counts)),
            "truncated_chunks": [{"chunk_id": cid, "tokens": n} for cid, n in sorted(over, key=lambda x: -x[1])],
            "query_max_length": self.encoder.query_max_length,
            "n_queries": len(queries),
            "n_queries_truncated": sum(n > self.encoder.query_max_length for n in q_counts),
            "max_query_tokens": max(q_counts) if q_counts else 0,
        }

    def save(self, index_dir: Path | None = None) -> None:
        index_dir = index_dir or INDEX_ROOT / self.name
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings)
        meta = {
            "spec": asdict(self.spec),
            "chunk_ids": self.chunk_ids,
            "corpus_fingerprint": self.fingerprint,
            "normalised": True,
            "dim": int(self.embeddings.shape[1]),
            "device": device_label(self.device),
            "timings": self.timings,
        }
        (index_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load_or_build(cls, spec: DenseSpec, corpus: list[CorpusChunk], device: str | None = None,
                      index_dir: Path | None = None) -> DenseRetriever:
        """Cached embeddings if they match ``spec`` and ``corpus``, else encode (and cache)."""
        device = device or default_device()
        index_dir = index_dir or INDEX_ROOT / spec.name
        meta_path = index_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["spec"] == asdict(spec) and meta["corpus_fingerprint"] == corpus_fingerprint(corpus):
                emb = np.load(index_dir / "embeddings.npy")
                return cls(spec, make_encoder(spec, device), emb, meta["chunk_ids"], meta["corpus_fingerprint"],
                           device)
        retriever = cls.build(spec, corpus, device)
        retriever.save(index_dir)
        return retriever

    def close(self) -> None:
        """Release the model(s) (frees GPU memory between runs)."""
        self.encoder = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
