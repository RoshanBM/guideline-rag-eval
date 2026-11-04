"""Dense retriever plumbing (FAISS search, normalisation, cache), with a fake encoder so no model
is downloaded. Model-specific behaviour (prefixes, MedCPT pair input) is checked by the spec."""

from __future__ import annotations

import numpy as np
import pytest

from src.retrieve.base import Retriever
from src.retrieve.corpus import CorpusChunk
from src.retrieve.dense import BGE_QUERY_PREFIX, DENSE_SPECS, DenseRetriever, DenseSpec, l2_normalise

VOCAB = ["amoxicillin", "blood", "pressure", "cough", "metformin"]


class FakeEncoder:
    """Bag-of-words vectors over VOCAB, deliberately unnormalised (lengths differ)."""

    doc_max_length = 8
    query_max_length = 4

    def _vec(self, text: str) -> np.ndarray:
        words = text.lower().split()
        return np.array([words.count(w) + 0.01 for w in VOCAB], dtype=np.float32)

    def encode_docs(self, chunks, batch_size):
        return np.stack([self._vec(c.text) for c in chunks])

    def encode_queries(self, queries):
        return np.stack([self._vec(q) for q in queries])

    def doc_token_counts(self, chunks):
        return [len(c.text.split()) + 2 for c in chunks]

    def query_token_counts(self, queries):
        return [len(q.split()) + 2 for q in queries]


SPEC = DenseSpec("fake", "st", "fake/model", "fake/model")
CORPUS = [
    CorpusChunk("amox", "d", "c", "table", "amoxicillin amoxicillin amoxicillin amoxicillin cough"),
    CorpusChunk("bp", "d", "c", "prose", "blood pressure"),
    CorpusChunk("met", "d", "c", "prose", "metformin"),
]


def make() -> DenseRetriever:
    enc = FakeEncoder()
    return DenseRetriever(SPEC, enc, enc.encode_docs(CORPUS, 2), [c.chunk_id for c in CORPUS], "fp", "cpu")


def test_search_is_cosine_similarity():
    r = make()
    assert isinstance(r, Retriever)
    hits = r.search("blood pressure", 3)
    assert [cid for cid, _ in hits][0] == "bp"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)  # identical direction
    assert all(-1.0001 <= s <= 1.0001 for _, s in hits)
    # "amoxicillin": the long chunk has 4x the raw dot product but cosine is what is returned
    assert r.search("amoxicillin", 1)[0] == ("amox", pytest.approx(0.97, abs=0.01))
    assert len(r.search("metformin", 2)) == 2 and r.search("x", 0) == []
    assert len(r.search("metformin", 50)) == 3  # k capped at corpus size


def test_embeddings_are_normalised():
    r = make()
    assert np.allclose(np.linalg.norm(r.embeddings, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(l2_normalise(np.zeros((1, 3))), axis=1), 0)  # no NaN


def test_truncation_report():
    rep = make().truncation_report(CORPUS, ["a b", "a b c d e f"])
    assert rep["n_chunks_truncated"] == 0 and rep["max_chunk_tokens"] == 7
    assert rep["n_queries_truncated"] == 1 and rep["max_query_tokens"] == 8


def test_specs_follow_the_model_cards():
    assert DENSE_SPECS["bge-small"].query_prefix == BGE_QUERY_PREFIX == DENSE_SPECS["bge-base"].query_prefix
    assert BGE_QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
    assert DENSE_SPECS["pubmedbert"].query_prefix == ""
    medcpt = DENSE_SPECS["medcpt"]
    assert (medcpt.query_model, medcpt.doc_model) == ("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder")
    assert (medcpt.query_max_length, medcpt.doc_max_length) == (64, 512)


def test_save_writes_embeddings_and_meta(tmp_path):
    r = make()
    r.save(tmp_path)
    assert np.allclose(np.load(tmp_path / "embeddings.npy"), r.embeddings)
    assert (tmp_path / "meta.json").exists()
