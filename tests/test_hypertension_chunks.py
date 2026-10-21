"""Sanity checks for ``data/chunks/hypertension.jsonl``.

Regenerate the file with ``python -m src.parse.who_hypertension`` before running.
Tests that need the raw PDF are skipped when it is not present.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from src.parse.schema import quote_verifies, validate_record
from src.parse.sources import SOURCES
from src.parse.text_utils import normalise_text

CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "chunks" / "hypertension.jsonl"
PDF_PATH = SOURCES["hypertension_pdf"].path

CONSTANTS = {
    "collection": "cardiometabolic",
    "doc": "WHO Guideline for the pharmacological treatment of hypertension in adults (2021)",
    "doc_id": "who-htn-2021",
    "source_url": "https://iris.who.int/handle/10665/344424",
    "licence": "CC BY-NC-SA 3.0 IGO",
    # AWaRe-only fields of the unified schema: always null here.
    "infection": None, "population": None, "population_raw": None,
    "context_header": None, "aware_category_note": None,
}
CHUNK_TYPES = {"recommendation", "prose", "table", "pico"}
STRENGTHS = {"strong", "conditional"}
CERTAINTY_RE = re.compile(r"^(high|moderate|low|very low)( to (high|moderate|low|very low))?$")
MAX_WORDS = 600

# Recommendation statements for which the PDF prints no strength or no certainty.
# Every grading line in section 3 was checked against the page renders: all 13
# statements print both, so this is empty. Add a chunk_id here (with the missing
# field and a comment citing the PDF page) only if the PDF genuinely prints none.
NO_PRINTED_GRADING: dict[str, set[str]] = {}

# Statements whose printed grading conflicts with another part of the source, and the
# reviewed resolution: chunk_id -> (strength_as_printed, resolved strength).
RESOLVED_DISCREPANCIES: dict[str, tuple[str, str]] = {
    # PDF p.19 (printed 7) prints "Strong recommendation, moderate- to high-certainty
    # evidence" beneath this statement. The Executive summary (PDF p.10, printed viii)
    # prints "Conditional recommendation"; the statement says "WHO suggests" (conditional
    # wording); and Al-Makki et al., Hypertension 2022;79:293–301, Figure 3 (data/raw/
    # hyp-79-293.pdf, PDF p.6) prints "Conditional recommendation". Resolved as conditional.
    "who-htn-2021-rec-01c": ("strong", "conditional"),
}


@pytest.fixture(scope="module")
def raw_lines() -> list[str]:
    if not CHUNKS_PATH.exists():
        pytest.fail(f"{CHUNKS_PATH} missing: run `python -m src.parse.who_hypertension` first")
    return CHUNKS_PATH.read_text(encoding="utf-8").splitlines()


@pytest.fixture(scope="module")
def chunks(raw_lines: list[str]) -> list[dict]:
    return [json.loads(line) for line in raw_lines]


@pytest.fixture(scope="module")
def by_id(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_every_line_matches_schema(chunks: list[dict]) -> None:
    for c in chunks:
        cid = c.get("chunk_id")
        assert validate_record(c) == [], cid  # unified schema, src/parse/schema.py
        for key, value in CONSTANTS.items():
            assert c[key] == value, (cid, key)
        # No parser-added text in this collection: the retrieval text is the source text.
        assert c["text"] == c["source_text"], cid
        assert re.fullmatch(r"who-htn-2021-[a-z0-9]+(-[a-z0-9]+)*", c["chunk_id"])
        assert all(p.strip() == p for p in c["section_path"])
        assert c["chunk_type"] in CHUNK_TYPES
        assert c["index_default"] is (c["chunk_type"] != "pico"), cid
        assert c["strength"] is None or c["strength"] in STRENGTHS
        assert c["certainty"] is None or CERTAINTY_RE.match(c["certainty"])
        assert c["strength_as_printed"] is None or c["strength_as_printed"] in STRENGTHS
        assert c["source_discrepancy"] is None or (isinstance(c["source_discrepancy"], str) and c["source_discrepancy"])
        if c["chunk_type"] == "recommendation":
            assert re.fullmatch(r"Recommendation \d+", c["rec_id"]), cid
            assert isinstance(c["rec_part"], int) and c["rec_part"] >= 1, cid
            assert c["parent_rec_id"] is None, cid
        else:
            for key in ("rec_id", "rec_part", "strength", "strength_as_printed", "certainty", "source_discrepancy"):
                assert c[key] is None, (cid, key)


def test_chunk_ids_unique(chunks: list[dict]) -> None:
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #
def test_every_statement_has_printed_strength_and_certainty(chunks: list[dict]) -> None:
    recs = [c for c in chunks if c["chunk_type"] == "recommendation"]
    assert len(recs) == 13
    missing = {
        c["chunk_id"]: {f for f in ("strength", "certainty") if c[f] is None}
        for c in recs
        if c["strength"] is None or c["certainty"] is None
    }
    assert missing == NO_PRINTED_GRADING


def test_source_discrepancies_are_recorded_and_resolved(chunks: list[dict], by_id: dict[str, dict]) -> None:
    flagged = {c["chunk_id"] for c in chunks if c["source_discrepancy"] is not None}
    assert flagged == set(RESOLVED_DISCREPANCIES)
    for cid, (printed, resolved) in RESOLVED_DISCREPANCIES.items():
        c = by_id[cid]
        assert (c["strength_as_printed"], c["strength"]) == (printed, resolved)
        note = c["source_discrepancy"]
        assert "printed p. 7" in note and "printed p. viii" in note
        assert "10.1161/HYPERTENSIONAHA.121.18192" in note
        assert note.endswith(f"Resolved as {resolved}.")
    # Without a recorded discrepancy, the stored strength is exactly what is printed.
    for c in chunks:
        if c["chunk_type"] == "recommendation" and c["source_discrepancy"] is None:
            assert c["strength"] == c["strength_as_printed"], c["chunk_id"]


def test_all_eight_recommendations_present(chunks: list[dict]) -> None:
    rec_ids = {c["rec_id"] for c in chunks if c["chunk_type"] == "recommendation"}
    assert rec_ids == {f"Recommendation {n}" for n in range(1, 9)}


def test_rec_parts_are_contiguous_and_in_order(chunks: list[dict]) -> None:
    parts: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        if c["chunk_type"] == "recommendation":
            parts[c["rec_id"]].append(c["rec_part"])
    for rid, seq in parts.items():
        assert seq == list(range(1, len(seq) + 1)), rid
    assert {rid: len(seq) for rid, seq in parts.items() if len(seq) > 1} == {
        "Recommendation 1": 3,
        "Recommendation 6": 3,
        "Recommendation 7": 2,
    }


def test_grading_line_matches_fields(chunks: list[dict]) -> None:
    """strength_as_printed/certainty must equal the grading line quoted at the end of the chunk."""
    for c in chunks:
        if c["chunk_type"] != "recommendation":
            continue
        last = c["text"].splitlines()[-1]
        m = re.match(r"^(Strong|Conditional) recommendation, (.+?)-certainty evidence$", last)
        assert m, (c["chunk_id"], last)
        assert c["strength_as_printed"] == m.group(1).lower()
        assert c["certainty"] == m.group(2).replace("- to", " to")


def test_implementation_remarks_point_to_their_recommendation(chunks: list[dict]) -> None:
    rec_by_section = {tuple(c["section_path"][:2]): c["rec_id"] for c in chunks if c["chunk_type"] == "recommendation"}
    remarks = [c for c in chunks if c["section_path"][-1] == "Implementation remarks"]
    # Recommendations 6 and 7 print no implementation remarks.
    assert {c["parent_rec_id"] for c in remarks} == {f"Recommendation {n}" for n in (1, 2, 3, 4, 5, 8)}
    for c in remarks:
        assert c["text"].startswith("Implementation remarks:")
        assert c["parent_rec_id"] == rec_by_section[tuple(c["section_path"][:2])], c["chunk_id"]
    for c in chunks:
        if c not in remarks:
            assert c["parent_rec_id"] is None, c["chunk_id"]


def test_pico_chunks_excluded_from_default_index(chunks: list[dict]) -> None:
    pico = [c for c in chunks if c["chunk_type"] == "pico"]
    assert len(pico) == 11
    assert all(c["section_path"][0].startswith("Annex 4") and not c["index_default"] for c in pico)
    assert all(c["index_default"] for c in chunks if c["chunk_type"] != "pico")


# --------------------------------------------------------------------------- #
# Size and pages
# --------------------------------------------------------------------------- #
def test_no_chunk_empty_or_too_long(chunks: list[dict]) -> None:
    for c in chunks:
        words = len(c["text"].split())
        assert 0 < words <= MAX_WORDS, (c["chunk_id"], words)


def test_pdf_page_numbers_within_pdf(chunks: list[dict]) -> None:
    if not PDF_PATH.exists():
        pytest.skip("raw PDF not available")
    import pymupdf

    page_count = pymupdf.open(PDF_PATH).page_count
    for c in chunks:
        assert 1 <= c["pdf_page_start"] <= c["pdf_page_end"] <= page_count, c["chunk_id"]


def test_printed_page_numbers(chunks: list[dict], by_id: dict[str, dict]) -> None:
    for c in chunks:
        # Every chunked page carries a printed number in this document.
        assert c["printed_page_start"] is not None and c["printed_page_end"] is not None, c["chunk_id"]
        span_pdf = c["pdf_page_end"] - c["pdf_page_start"]
        assert c["printed_page_end"] - c["printed_page_start"] == span_pdf, c["chunk_id"]
    # Spot checks read off the page-number tabs of the rendered pages.
    assert (by_id["who-htn-2021-rec-01a"]["pdf_page_start"], by_id["who-htn-2021-rec-01a"]["printed_page_start"]) == (19, 7)
    assert (by_id["who-htn-2021-annex-4-pico-01"]["pdf_page_start"], by_id["who-htn-2021-annex-4-pico-01"]["printed_page_start"]) == (56, 44)


# --------------------------------------------------------------------------- #
# Text quality and quote matching
# --------------------------------------------------------------------------- #
def test_text_is_normalised_and_clean(chunks: list[dict]) -> None:
    for c in chunks:
        text = c["text"]
        assert normalise_text(text) == text, c["chunk_id"]
        assert not re.search("[-]", text), c["chunk_id"]  # unresolved PUA glyphs
        assert "GUIDELINE FOR THE PHARMACOLOGICAL TREATMENT" not in text, c["chunk_id"]  # running header
        assert not re.search(r"(?m)^\[", text), c["chunk_id"]  # unconverted bullet glyph


def test_quotes_round_trip(chunks: list[dict]) -> None:
    """Any exact substring of a chunk's source text must verify against the chunk."""
    for c in chunks:
        text = c["source_text"]
        for start in range(0, len(text), max(1, len(text) // 5)):
            quote = text[start : start + 80]
            if quote.strip():
                assert quote_verifies(quote, c), (c["chunk_id"], quote)


TABLE_QUOTE = (
    "High | We are very confident that the true effect lies close to that of the estimate of effect.\n"
    "Moderate | We are moderately confident in the effect estimate."
)
BULLET_QUOTE = (
    "Implementation remarks:\n"
    "• Initiation of pharmacological hypertension (HTN) treatment should start no later than four weeks "
    "following diagnosis of HTN."
)


def test_quote_from_table_chunk_with_pipes(by_id: dict[str, dict]) -> None:
    chunk = by_id["who-htn-2021-table-1"]
    assert "|" in chunk["source_text"] and TABLE_QUOTE in chunk["source_text"]  # the quote really is verbatim
    assert quote_verifies(TABLE_QUOTE, chunk)
    # The same quote as a model would typically emit it: line break flattened to a space.
    assert quote_verifies(TABLE_QUOTE.replace("\n", " "), chunk)
    # Paraphrases and altered wording must not match.
    assert not quote_verifies("High | We are very sure that the true effect is close to the estimate of effect.", chunk)
    assert not quote_verifies("High: We are very confident that the true effect lies close to that of the estimate of effect.", chunk)


def test_quote_from_bulleted_chunk(by_id: dict[str, dict]) -> None:
    chunk = by_id["who-htn-2021-s3-1-implementation-remarks"]
    assert "•" in chunk["source_text"] and BULLET_QUOTE in chunk["source_text"]
    assert quote_verifies(BULLET_QUOTE, chunk)
    assert quote_verifies(BULLET_QUOTE.replace("\n", " "), chunk)
    # The bullet item quoted without its marker is still a verbatim substring.
    assert quote_verifies("Initiation of pharmacological hypertension (HTN) treatment should start", chunk)
    # Paraphrase, and a quote with one number changed, are rejected.
    assert not quote_verifies("• HTN treatment should begin within four weeks of the diagnosis of HTN.", chunk)
    assert not quote_verifies("If blood pressure level is high (e.g. systolic ≥150 mmHg or diastolic ≥100 mmHg)", chunk)


def test_known_content_survives(by_id: dict[str, dict], chunks: list[dict]) -> None:
    """Spot checks against text read directly off the PDF pages."""
    rec1a = by_id["who-htn-2021-rec-01a"]
    assert "systolic blood pressure of ≥140 mmHg or diastolic blood pressure of ≥90 mmHg" in rec1a["text"]
    assert "confirmed diagnosis" in rec1a["text"]  # "fi" ligature resolved
    assert (rec1a["strength"], rec1a["certainty"], rec1a["rec_part"]) == ("strong", "moderate to high", 1)
    fig5 = next(c for c in chunks if "Fig. 5 Algorithm 1" in c["text"])
    assert "Telmisartan 40–Amlodipine 5 mg** once a day" in fig5["text"]
    assert "hydrochlorothiazide 25 mg or chlorthalidone 12.5 mg once a day" in fig5["text"]
    assert any("conflicts of interest" in c["text"] for c in chunks)  # "fl" ligature resolved


def test_parser_is_deterministic(raw_lines: list[str]) -> None:
    if not PDF_PATH.exists():
        pytest.skip("raw PDF not available")
    from src.parse.who_hypertension import parse

    first = [json.dumps(c.to_record(), ensure_ascii=False) for c in parse()[0]]
    second = [json.dumps(c.to_record(), ensure_ascii=False) for c in parse()[0]]
    assert first == second == raw_lines
