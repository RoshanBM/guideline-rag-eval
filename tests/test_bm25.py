"""Tests for the retrieval corpus, the BM25 tokenizer and the BM25 retriever."""

from __future__ import annotations

import pytest

from src.evaluate.relevance import load_chunks
from src.retrieve.base import Retriever
from src.retrieve.bm25 import BM25Retriever, StaleIndexError
from src.retrieve.corpus import CorpusChunk, load_corpus
from src.retrieve.tokenize import tokenize


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("text", "expected"), [
    ("500 mg", ["500", "mg"]),
    ("500mg", ["500", "mg"]),
    ("≥130", ["≥130", "130"]),
    ("≥ 130 mmHg", ["≥130", "130", "mmhg"]),
    (">=140", ["≥140", "140"]),
    ("<2+", ["<2", "2+", "2"]),
    ("3–5 days", ["3", "5", "days"]),
    ("0.9% sodium chloride", ["0.9", "sodium", "chloride"]),
    ("HbA1c", ["hba1c"]),
    ("Do not give antibiotics", ["not", "give", "antibiotics"]),
    ("No antibiotic is needed", ["no", "antibiotic", "needed"]),
    ("Don't treat; you can't", ["not", "treat", "not"]),
    ("Table 6.5", ["table", "6.5"]),
    ("2 x daily", ["2", "x", "daily"]),
    ("5d", ["5"]),
    ("", []),
])
def test_tokenize_unstemmed(text, expected):
    assert tokenize(text, stem=False) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("Antibiotics are not needed", ["antibiot", "not", "need"]),
    ("antibiotic", ["antibiot"]),
    ("treating treatment treated", ["treat", "treatment", "treat"]),
    ("3–5 days", ["3", "5", "day"]),
    ("≥130 mmHg", ["≥130", "130", "mmhg"]),
    ("<2+ ketones", ["<2", "2+", "2", "keton"]),
    ("no, nor, not", ["no", "nor", "not"]),
    ("0.9% sodium chloride", ["0.9", "sodium", "chlorid"]),
])
def test_tokenize_stemmed_is_the_default(text, expected):
    assert tokenize(text) == expected
    assert tokenize(text, stem=True) == expected


def test_numbers_are_never_stopwords():
    assert tokenize("a 1 the 2 of 10") == ["1", "2", "10"]


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
def test_header_and_body_splits_retrieval_text():
    corpus = load_corpus()
    with_header = [c for c in corpus if c.context_header]
    assert with_header, "AWaRe and HEARTS-D chunks have context headers"
    for c in corpus:
        header, body = c.header_and_body()
        assert (header + "\n" + body if header else body) == c.text


def test_corpus_indexes_only_default_chunks_and_uses_text():
    records = load_chunks()
    corpus = load_corpus()
    defaults = [r for r in records if r["index_default"]]
    assert [c.chunk_id for c in corpus] == [r["chunk_id"] for r in defaults]
    assert all(c.chunk_type != "pico" for c in corpus)
    by_id = {r["chunk_id"]: r for r in defaults}
    assert all(c.text == by_id[c.chunk_id]["text"] for c in corpus)


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
def toy_corpus(extra: str = "") -> list[CorpusChunk]:
    texts = {
        "amox": "Amoxicillin 500 mg orally three times daily for 5 days",
        "nitro": "Nitrofurantoin 100 mg orally twice daily for 5 days",
        "bp": "Start treatment if systolic blood pressure ≥140 mmHg" + extra,
        "noabx": "Antibiotics are not needed for acute bronchitis",
        "dup1": "identical text about cough",
        "dup2": "identical text about cough",
    }
    return [CorpusChunk(cid, "doc", "coll", "prose", t) for cid, t in texts.items()]


@pytest.mark.parametrize("stem", [True, False])
def test_search_ranks_matching_chunk_first(stem):
    r = BM25Retriever.build(toy_corpus(), stem=stem)
    assert isinstance(r, Retriever)
    assert r.name == ("bm25" if stem else "bm25-nostem")
    assert r.search("amoxicillin dose", 3)[0][0] == "amox"
    assert r.search("blood pressure 140", 3)[0][0] == "bp"
    assert r.search("are antibiotics needed for bronchitis?", 1)[0][0] == "noabx"


def test_search_returns_only_matches_and_respects_k():
    r = BM25Retriever.build(toy_corpus())
    hits = r.search("mg orally daily", 10)
    assert {cid for cid, _ in hits} == {"amox", "nitro"}
    assert all(score > 0 for _, score in hits)
    assert len(r.search("mg orally daily", 1)) == 1
    assert r.search("zzzz unknownword", 5) == []
    assert r.search("the of and", 5) == []  # only stopwords
    assert r.search("amoxicillin", 0) == []


def test_ties_keep_corpus_order():
    r = BM25Retriever.build(toy_corpus())
    hits = r.search("cough", 5)
    assert [cid for cid, _ in hits] == ["dup1", "dup2"]
    assert hits[0][1] == hits[1][1]


def test_stemming_matches_inflected_forms():
    assert BM25Retriever.build(toy_corpus(), stem=True).search("antibiotic need", 1)[0][0] == "noabx"
    # unstemmed: "antibiotic" and "need" do not match "Antibiotics ... needed"
    assert BM25Retriever.build(toy_corpus(), stem=False).search("antibiotic need", 1) == []


@pytest.mark.parametrize("stem", [True, False])
def test_save_load_round_trip(tmp_path, stem):
    corpus = toy_corpus()
    built = BM25Retriever.build(corpus, stem=stem)
    built.save(tmp_path)
    loaded = BM25Retriever.load(tmp_path, corpus, stem=stem)
    assert loaded.name == built.name
    for q in ("amoxicillin 500 mg", "≥140 mmHg", "bronchitis antibiotics not needed"):
        assert loaded.search(q, 6) == pytest.approx(built.search(q, 6))


def test_load_refuses_stale_index(tmp_path):
    BM25Retriever.build(toy_corpus()).save(tmp_path)
    with pytest.raises(StaleIndexError):
        BM25Retriever.load(tmp_path, toy_corpus(extra=" changed"))
    with pytest.raises(StaleIndexError, match="stem"):
        BM25Retriever.load(tmp_path, toy_corpus(), stem=False)
    rebuilt = BM25Retriever.load_or_build(toy_corpus(extra=" changed"), index_dir=tmp_path)
    assert BM25Retriever.load(tmp_path, toy_corpus(extra=" changed")).chunk_ids == rebuilt.chunk_ids
