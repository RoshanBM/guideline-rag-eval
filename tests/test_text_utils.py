"""Unit tests for the shared text normalisation."""

from __future__ import annotations

from src.parse.text_utils import (
    apply_corrections, build_vocabulary, join_lines, normalise_for_match, normalise_text, quote_in_text,
)


def test_normalise_text_is_idempotent_and_tidies_whitespace() -> None:
    raw = "  Signiﬁcant  effect here \n\n\n next­line  "
    once = normalise_text(raw)
    assert once == "Significant effect here\nnextline"
    assert normalise_text(once) == once


def test_combining_accents_are_composed() -> None:
    assert normalise_text("Médecins Sans Frontières") == "Médecins Sans Frontières"


def test_join_lines_keeps_compound_hyphens_by_default() -> None:
    assert join_lines(["a cost-", "effective option"]) == "a cost-effective option"


def test_join_lines_removes_typesetting_hyphen_when_word_attested() -> None:
    vocab = build_vocabulary(["this is significant"])
    assert join_lines(["signi-", "ficant"], vocab) == "significant"


def test_join_lines_tight_dash_and_spaced_dash() -> None:
    assert join_lines(["Telmisartan 40–", "Amlodipine 5 mg"]) == "Telmisartan 40–Amlodipine 5 mg"
    assert join_lines(["Hypertension –", "or elevated"]) == "Hypertension – or elevated"


def test_quote_matching_folds_typography_but_not_case() -> None:
    chunk = "WHO recommends a target of <140/90 mmHg in patients’ care – see “Table 1”."
    assert quote_in_text("patients' care - see \"Table 1\"", chunk)
    assert quote_in_text("a target of\n<140/90 mmHg", chunk)
    assert not quote_in_text("who recommends", chunk)
    assert normalise_for_match("a\n b") == "a b"


def test_corrections_map_applies_whole_words_only() -> None:
    text, applied = apply_corrections("give 20–50 mI of 0.9% NaCI; Iimited; unable to to ingest; limb iscaemia")
    assert text == "give 20–50 ml of 0.9% NaCl; limited; unable to ingest; limb ischaemia"
    assert [(e["printed"], e["corrected"], e["count"]) for e in applied] == [
        ("mI", "ml", 1), ("NaCI", "NaCl", 1), ("Iimited", "limited", 1), ("to to", "to", 1), ("iscaemia", "ischaemia", 1),
    ]
    # Correct words and other words containing the patterns are untouched.
    assert apply_corrections("mIU/L, NaCl, limited, ischaemia, go to town") == (
        "mIU/L, NaCl, limited, ischaemia, go to town", [])


def test_quote_matching_treats_corrections_as_equivalent() -> None:
    printed = "give 20–50 mI of 50% glucose; drip 0.9% NaCI; unable to to ingest"
    corrected = "give 20–50 ml of 50% glucose; drip 0.9% NaCl; unable to ingest"
    assert quote_in_text("20–50 ml of 50% glucose", printed)  # corrected quote, printed text
    assert quote_in_text("20–50 mI of 50% glucose", corrected)  # printed quote, corrected text
    assert quote_in_text("unable to ingest", printed) and quote_in_text("unable to to ingest", corrected)
    assert not quote_in_text("20–50 mg of 50% glucose", printed)
