"""Tests for src/generate/verify.py: citation checks and number + unit flags, on real chunks."""

from __future__ import annotations

import pytest

from src.evaluate.relevance import load_chunks
from src.parse.schema import verifiable_text
from src.generate.verify import (
    check_result,
    clean_quote,
    number_units,
    supported_set,
    unlabelled_number_sentences,
    unsupported_numbers,
    verify_citations,
)

FIG1 = "who-hearts-d-2020-fig-1"


@pytest.fixture(scope="module")
def chunks() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in load_chunks()}


def with_header_and_flow(chunks):
    """A HEARTS-D chunk with a context header, and the Fig. 1 flowchart (flow_text)."""
    header_chunk = next(c for c in chunks.values() if c["doc_id"] == "who-hearts-d-2020" and c["context_header"]
                        and c["chunk_type"] == "prose")
    return header_chunk, chunks[FIG1]


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #
def test_verbatim_quote_from_source_text_verifies(chunks):
    fig1 = chunks[FIG1]
    checks = verify_citations([{"label": "S1", "quote": "GIVE METFORMIN 1000 mg 2 x daily"}], {"S1": fig1})
    assert checks == [{"label": "S1", "quote": "GIVE METFORMIN 1000 mg 2 x daily", "chunk_id": FIG1,
                       "status": "verified", "verified": True, "tolerant": False, "correct_label": None,
                       "correct_chunk_id": None, "reason": None}]


def test_whitespace_dash_and_quote_marks_are_folded(chunks):
    fig1 = chunks[FIG1]
    checks = verify_citations([{"label": "S1", "quote": '  "GIVE  METFORMIN\n1000 mg 2 x daily"  '}], {"S1": fig1})
    assert checks[0]["verified"]
    assert clean_quote("“REVIEW in 3–5 days”") == "REVIEW in 3–5 days"


def test_paraphrase_case_change_and_wrong_label_fail(chunks):
    fig1 = chunks[FIG1]
    other = next(c for c in chunks.values() if c["doc_id"] == "who-aware-2022")
    checks = verify_citations([
        {"label": "S1", "quote": "give metformin 1000 mg 2 x daily"},   # case changed beyond the first letter
        {"label": "S1", "quote": "GIVE METFORMIN 1000 mg twice daily"},  # paraphrase
        {"label": "S2", "quote": "GIVE METFORMIN 1000 mg 2 x daily"},   # real quote, wrong source
        {"label": "S9", "quote": "GIVE METFORMIN"},                      # label not shown: matches S1
        {"label": "S9", "quote": "no such words"},                       # label not shown, no match
        {"label": "S1", "quote": '""'},                                  # empty after cleaning
    ], {"S1": fig1, "S2": other})
    assert [c["verified"] for c in checks] == [False] * 6
    assert [c["status"] for c in checks] == ["not_verified", "not_verified", "verified_wrong_label",
                                             "verified_wrong_label", "not_verified", "not_verified"]
    assert [c["reason"] for c in checks] == ["not in source text", "not in source text", "matches S1, not S2",
                                             "matches S1, not S9", "unknown label", "empty quote"]
    assert (checks[2]["correct_label"], checks[2]["correct_chunk_id"]) == ("S1", FIG1)


@pytest.mark.parametrize(("quote", "status", "tolerant"), [
    ("Antibiotics are not needed for most cases since most cases are self-limiting", "verified", False),
    ("antibiotics are not needed for most cases since most cases are self-limiting", "verified", True),  # 1st letter
    ("Antibiotics are not needed for most cases.", "verified", True),         # added final full stop
    ("Antibiotics are not needed for most cases...", "verified", True),       # trailing ellipsis
    ("Antibiotics are not needed for most cases …", "verified", True),        # trailing … (with a space)
    ("antibiotics are not needed for most cases.", "verified", True),         # both at once
    ("Antibiotics are not needed ... self-limiting", "not_verified", False),  # inner ellipsis
    ("ANTIBIOTICS are not needed", "not_verified", False),                    # case beyond the first letter
    ("Antibiotics are not needed for most cases [S1]", "not_verified", False),  # label inside the quote
    ("Antibiotics are not needed for many cases.", "not_verified", False),    # changed word
])
def test_edge_tolerance(chunks, quote, status, tolerant):
    rec = chunks["who-aware-2022-ch06-noabx-key-messages"]
    assert "Antibiotics are not needed for most cases since most cases are self-limiting" in rec["source_text"]
    check = verify_citations([{"label": "S1", "quote": quote}], {"S1": rec})[0]
    assert (check["status"], check["tolerant"]) == (status, tolerant)


def test_wrong_label_uses_the_tolerance_too(chunks):
    fig1, other = chunks[FIG1], chunks["who-aware-2022-ch06-noabx-key-messages"]
    check = verify_citations([{"label": "S1", "quote": "antibiotics are not needed for most cases."}],
                             {"S1": fig1, "S2": other})[0]
    assert (check["status"], check["correct_label"], check["tolerant"]) == ("verified_wrong_label", "S2", True)


@pytest.mark.parametrize(("answer", "unlabelled"), [
    ("Give 500 mg every 8 hours [S2]. Review in 3 days [S1].", []),
    ("Give 500 mg every 8 hours. [S2] Review after a week.", []),  # label opening the next sentence
    ("Give 500 mg every 8 hours. Antibiotics are not needed [S1].", ["Give 500 mg every 8 hours."]),
    ("See [S3] and [S4]. Target <140/90 mmHg.", ["Target <140/90 mmHg."]),  # digits in labels do not count
    ("Most cases are viral [S1, S3]; no antibiotic is needed.", []),
    ("", []),
])
def test_unlabelled_number_sentences(answer, unlabelled):
    assert unlabelled_number_sentences(answer) == unlabelled


def test_context_header_and_flow_text_never_verify(chunks):
    header_chunk, fig1 = with_header_and_flow(chunks)
    header_quote = header_chunk["context_header"].rstrip(":")
    flow_quote = fig1["flow_text"].split(";")[0].strip()[:60]
    assert flow_quote and flow_quote not in fig1["source_text"]
    checks = verify_citations([{"label": "S1", "quote": header_quote}, {"label": "S2", "quote": flow_quote}],
                              {"S1": header_chunk, "S2": fig1})
    assert [c["verified"] for c in checks] == [False, False]


def test_footnotes_verify(chunks):
    chunk = chunks["who-aware-2022-ch06-dosing-adults-first-choice"]
    quote = "In settings with a low prevalence of rheumatic fever"
    assert quote not in chunk["source_text"]  # it is a footnote, kept only in text
    assert verify_citations([{"label": "S1", "quote": quote}], {"S1": chunk})[0]["verified"]


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("text", "expected"), [
    ("500 mg given every 8 hours for 3–5 days", ["500 mg", "8 hour", "3-5 day"]),
    ("GLICLAZIDE 80 g 2 x daily", ["80 g"]),
    ("≥140 mmHg; 130 to 139 mmHg", ["140 mmhg", "130-139 mmhg"]),
    ("glucose 50% 20–50 mI", ["50 %", "20-50 ml"]),  # documented typo mI -> ml
    ("1,000 mg or 2.5 mg", ["1000 mg", "2.5 mg"]),
    ("500 mg (800 000 IU) given every 6 hours", ["500 mg", "800000 iu", "6 hour"]),  # space thousands
    ("80 given, 2 x daily, eGFR 25", []),
])
def test_number_units(text, expected):
    assert number_units(text) == expected


def test_source_ranges_support_their_endpoints():
    assert supported_set("for 5–7 days") == {"5-7 day", "5 day", "7 day"}


def test_unsupported_numbers(chunks):
    fig1 = chunks[FIG1]
    answer = ('Give metformin "1000 mg 2 x daily", review in 5 days, gliclazide 40 mg for 7 days '
              "and 3-5 days; a 12 kg child")
    flags = unsupported_numbers(answer, [fig1], question="What dose for a 12 kg child?")
    values = {f["value"]: f["in_question"] for f in flags}
    assert "1000 mg" not in values  # in the source
    assert "5 day" not in values and "3-5 day" not in values  # source prints "3–5 days"
    assert values["40 mg"] is False and values["7 day"] is False  # in no part of Fig. 1
    assert values["12 kg"] is True  # only restates the question
    assert unsupported_numbers("80 g", [fig1]) == []  # printed in the source, so not "unsupported"
    # a bare number in the question ("about 134") covers the answer's "134 mmHg"
    assert unsupported_numbers("134 mmHg", [fig1], question="systolic pressure sits at about 134. Treat?") == [
        {"value": "134 mmhg", "in_question": True}]
    assert unsupported_numbers("135 mmHg", [fig1], question="about 134?")[0]["in_question"] is False
    assert unsupported_numbers("Give 5 mg.", []) == [{"value": "5 mg", "in_question": False}]


def test_check_result_uses_cited_chunks_only(chunks):
    fig1 = chunks[FIG1]
    other_id = "who-aware-2022-ch06-dosing-adults-first-choice"
    only_in_s2 = sorted(supported_set(verifiable_text(chunks[other_id])) - supported_set(verifiable_text(fig1)))
    value = next(v for v in only_in_s2 if "-" not in v.split(" ")[0])
    result = {
        "question": "q", "valid": True, "refused": False,
        "retrieved": [{"label": "S1", "chunk_id": FIG1, "score": 0.8}, {"label": "S2", "chunk_id": other_id, "score": 0.7}],
        "answer": f'Give "GIVE METFORMIN 1000 mg 2 x daily" [S1] or {value}.',
        "citations": [{"label": "S1", "quote": "GIVE METFORMIN 1000 mg 2 x daily"}],
    }
    out = check_result(result, chunks)
    assert [c["verified"] for c in out["citation_checks"]] == [True]
    # the value is printed in S2, but S2 is not cited
    assert [f["value"] for f in out["number_flags"]] == [value]
    assert out["unlabelled_number_sentences"] == []
    assert check_result({**result, "refused": True}, chunks) == {"citation_checks": [], "number_flags": [],
                                                                  "unlabelled_number_sentences": []}


def test_check_result_uses_the_sources_shown_and_the_matching_source_for_numbers(chunks):
    """Dose-route results show other chunks than retrieved; a wrong-label quote's numbers count
    for the source it matches."""
    adults = chunks["who-aware-2022-ch06-dosing-adults-first-choice"]
    result = {
        "question": "q", "valid": True, "refused": False,
        "retrieved": [{"label": "S1", "chunk_id": FIG1, "score": 0.8}],
        "sources": [{"label": "S1", "chunk_id": FIG1, "score": None},
                    {"label": "S2", "chunk_id": adults["chunk_id"], "score": None}],
        "answer": "Amoxicillin 500 mg every 8 hours [S1].",
        "citations": [{"label": "S1", "quote": "Amoxicillin(oral): 500 mg given every 8 hours"}],
    }
    out = check_result(result, chunks)
    assert out["citation_checks"][0]["status"] == "verified_wrong_label"
    assert out["citation_checks"][0]["correct_label"] == "S2"
    assert out["number_flags"] == []  # 500 mg and 8 hours are in S2
