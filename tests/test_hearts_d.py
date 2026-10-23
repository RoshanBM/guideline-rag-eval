"""Sanity checks for ``data/chunks/hearts_d.jsonl`` (WHO HEARTS-D, type 2 diabetes).

Regenerate the file with ``python -m src.parse.who_hearts_d`` before running.
Tests that need the raw PDF are skipped when it is not present.

HEARTS-D has no medicine dose *tables* (doses appear only in flowcharts and prose), so the
table-value check is made against the PDF itself: every number printed inside a table or
flowchart must appear in its chunk. The dose statements are exported separately to
``data/interim/hearts_d_doses.csv`` and checked against the chunks here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pymupdf
import pytest

from src.parse.pdf_utils import PUA_RE
from src.parse.schema import quote_verifies, validate_record, verifiable_text
from src.parse.sources import SOURCES
from src.parse.text_utils import TEXT_CORRECTIONS, apply_corrections, normalise_for_match
from src.parse.who_hearts_d import BLACK, BOX_GREY, GREY, WHITE  # measured colours of the source PDF

CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "chunks" / "hearts_d.jsonl"
PDF_PATH = SOURCES["hearts_d_pdf"].path

CONSTANTS = {
    "collection": "cardiometabolic",
    "doc_id": "who-hearts-d-2020",
    "source_url": "https://www.who.int/publications/i/item/WHO-UCN-NCD-20.1",
    "licence": "CC BY-NC-SA 3.0 IGO",
    # Fields that do not apply to this document: always null.
    "infection": None, "population": None, "population_raw": None, "rec_id": None,
    "rec_part": None, "parent_rec_id": None, "strength": None, "strength_as_printed": None,
    "certainty": None, "aware_category_note": None,
}
CHUNK_TYPES = {"prose", "table", "flowchart"}
MAX_WORDS = 600
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
FOOTER_TOP = 790.0


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        pytest.fail(f"{CHUNKS_PATH} missing: run `python -m src.parse.who_hearts_d` first")
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def by_id(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


needs_pdf = pytest.mark.skipif(not PDF_PATH.exists(), reason="raw HEARTS-D PDF not present")


# --------------------------------------------------------------------------- #
# Schema, IDs, sizes
# --------------------------------------------------------------------------- #


def test_schema(chunks: list[dict]) -> None:
    assert chunks
    for c in chunks:
        assert validate_record(c) == [], c["chunk_id"]
        for field, value in CONSTANTS.items():
            assert c[field] == value, (c["chunk_id"], field)
        assert c["chunk_type"] in CHUNK_TYPES, c["chunk_id"]
        assert c["index_default"] is True
        assert c["chunk_id"].startswith("who-hearts-d-2020-")
        assert c["context_header"] and c["context_header"].startswith("HEARTS-D - ")


def test_unique_ids(chunks: list[dict]) -> None:
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), [i for i, n in Counter(ids).items() if n > 1]


def test_sizes(chunks: list[dict]) -> None:
    for c in chunks:
        words = len(c["text"].split())
        assert 0 < words <= MAX_WORDS, (c["chunk_id"], words)
        assert c["source_text"].strip()


def test_expected_inventory(chunks: list[dict]) -> None:
    """All eight tables and the three flowcharts (Fig. 2 split by column) are present."""
    tables = sorted(int(c["chunk_id"].rsplit("-", 1)[1]) for c in chunks if c["chunk_type"] == "table")
    assert tables == list(range(1, 9))
    flowcharts = {c["chunk_id"] for c in chunks if c["chunk_type"] == "flowchart"}
    assert flowcharts == {
        "who-hearts-d-2020-fig-1",
        "who-hearts-d-2020-fig-2-part1",
        "who-hearts-d-2020-fig-2-part2",
        "who-hearts-d-2020-fig-2-part3",
        "who-hearts-d-2020-annex1-flowchart",
    }


def test_pages(chunks: list[dict]) -> None:
    """Printed page numbers come from the footers; in this PDF they equal the PDF page."""
    for c in chunks:
        assert 8 <= c["pdf_page_start"] <= c["pdf_page_end"] <= 35, c["chunk_id"]
        assert c["printed_page_start"] == c["pdf_page_start"], c["chunk_id"]
        assert c["printed_page_end"] == c["pdf_page_end"], c["chunk_id"]
    pages = {p for c in chunks for p in range(c["pdf_page_start"], c["pdf_page_end"] + 1)}
    # p. 28 is the reference list; every other content page contributes text.
    assert pages == set(range(8, 36)) - {28}


@needs_pdf
def test_determinism(chunks: list[dict]) -> None:
    from src.parse.who_hearts_d import parse

    first = [c.to_record() for c in parse()[0]]
    second = [c.to_record() for c in parse()[0]]
    assert first == second
    assert first == chunks, "data/chunks/hearts_d.jsonl is stale: re-run python -m src.parse.who_hearts_d"


# --------------------------------------------------------------------------- #
# Text fidelity
# --------------------------------------------------------------------------- #


def test_text_structure(chunks: list[dict]) -> None:
    """text = header + body (flow_text for flowcharts, else source_text) + notes, corrected."""
    for c in chunks:
        body = apply_corrections(c["flow_text"] or c["source_text"])[0]
        assert c["text"].startswith(c["context_header"] + "\n" + body), c["chunk_id"]
        assert (c["flow_text"] is not None) == (c["chunk_type"] == "flowchart"), c["chunk_id"]
        for t in (c["text"], c["source_text"]):
            assert not PUA_RE.search(t), c["chunk_id"]
            assert "�" not in t and "••" not in t, c["chunk_id"]  # no doubled bullets
            assert not re.search("[ﬀ-ﬆ]", t), c["chunk_id"]  # ligatures expanded


def test_quotes_verify(chunks: list[dict]) -> None:
    """Every source line (and every note) verifies as a quote; the parser-written header does not."""
    for c in chunks:
        for line in verifiable_text(c).split("\n"):
            assert quote_verifies(line, c), (c["chunk_id"], line)
        assert not quote_verifies(c["context_header"], c), c["chunk_id"]


def test_ligatures_and_spaces(chunks: list[dict]) -> None:
    """Words around ligatures read correctly (they were the hard cases in the earlier browser re-print)."""
    text = "\n".join(c["source_text"] for c in chunks)
    for phrase in (
        "Diagnostic cut-off values are presented in Table 3",
        "primary health care staff and trainers",
        "plasma glucose cut-off point for hypoglycaemia",
        "a highly specific microvascular",
        "intra-individual variability",
        "Intensification of treatment when metformin alone fails",
    ):
        assert phrase in text, phrase


def test_superscripts(by_id: dict[str, dict]) -> None:
    """Footnote markers are removed from the text (the footnote goes under Notes); the
    exponent of 'm²' is kept."""
    kidney = by_id["who-hearts-d-2020-ch3-diabetic-kidney-disease"]
    assert "Estimated glomerular filtration rate (eGFR) of <60 mL/min per 1.73 m² on at least" in kidney["source_text"]
    assert "1 eGFR is calculated from serum creatinine" in kidney["text"].split("\nNotes:\n")[1]
    protocol = by_id["who-hearts-d-2020-ch2-blood-glucose-management-protocol-p2"]
    assert "add human insulin to oral medication (Annex 1)." in protocol["source_text"]
    assert "If insulin is unsuitable, a DPP-4 inhibitor" in protocol["source_text"]
    notes = protocol["text"].split("\nNotes:\n")[1]
    assert notes.startswith("1 Insulin analogues are not recommended")
    assert "\n2 Insulin treatment could be unsuitable" in notes


def test_skipped_content_absent(chunks: list[dict]) -> None:
    text = "\n".join(c["text"] for c in chunks)
    for skipped in (
        "Acknowledgements", "WHO expresses special appreciation",  # p. 7
        "angiotensin-converting enzyme\nACR",  # abbreviations list
        "Some rights reserved",  # copyright page
        "Levey AS", "Global report on diabetes",  # 6 Resources, p. 28
        "International Working Group on Diabetic Foot",  # Annex 2 Resources, p. 35
        "Fig. 5 Dorsal pedis palpation", "Fig. 11 Appropriate shoe shape",  # photograph captions
        "HEARTS – D Diagnosis and management of Type 2 diabetes",  # running footer
    ):
        assert skipped not in text, skipped


def test_side_boxes_kept(by_id: dict[str, dict]) -> None:
    """Side boxes are kept as their own block (title line, then one bullet per item)."""
    intro = by_id["who-hearts-d-2020-intro-p2"]["source_text"]
    assert "Risk factors for type 2 diabetes\n• overweight/obesity\n• physical inactivity\n• age\n" in intro
    eye = by_id["who-hearts-d-2020-ch3-microvascular-complications"]["source_text"]
    # The box beside a bullet is placed after that bullet, not inside it.
    assert "It is among the leading causes of blindness.\nRisk factors for diabetic retinopathy\n" in eye


# --------------------------------------------------------------------------- #
# Tables and flowcharts against the PDF
# --------------------------------------------------------------------------- #


def _numbers(text: str) -> Counter:
    return Counter(NUMBER_RE.findall(text))


def _page_spans(page: pymupdf.Page) -> list[dict]:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip() and s["bbox"][1] < FOOTER_TOP:
                    spans.append(s)
    return spans


@needs_pdf
def test_table_values_in_chunks(chunks: list[dict]) -> None:
    """Every number printed in a table body or its notes appears in the table chunk.

    The table body is read independently of the parser: the 9 pt black/white spans below the
    caption, up to the next text of another style; notes are the grey 9 pt lines starting
    with "*" that follow.
    """
    doc = pymupdf.open(PDF_PATH)
    tables = [c for c in chunks if c["chunk_type"] == "table"]
    checked = 0
    for c in tables:
        page = doc[c["pdf_page_start"] - 1]
        caption = c["source_text"].split("\n", 1)[0]
        number = caption.split()[1]
        spans = sorted(_page_spans(page), key=lambda s: (round(s["bbox"][1], 1), s["bbox"][0]))
        start = next(s for s in spans if s["text"].strip().startswith(f"Table {number}"))["bbox"][3]
        expected: Counter = Counter()
        in_table = True
        for s in spans:
            if s["bbox"][1] <= start or s["color"] == BOX_GREY:
                continue
            body = s["size"] <= 9.6 and s["color"] in (BLACK, WHITE)
            note = s["color"] == GREY and s["size"] <= 9.5
            if in_table and body:
                expected.update(_numbers(s["text"]))
                continue
            in_table = False
            if not note:
                break
            if s["size"] >= 7:  # not a page-footnote number
                expected.update(_numbers(s["text"]))
        got = _numbers(verifiable_text(c))
        missing = expected - got
        assert not missing, (c["chunk_id"], missing)
        checked += sum(expected.values())
    assert checked > 50  # Tables 2 and 4 contain no numbers; the others contain many


@needs_pdf
@pytest.mark.parametrize("pdf_page", [16, 25, 29])
def test_flowchart_values_in_chunks(chunks: list[dict], pdf_page: int) -> None:
    """Every number printed on a flowchart page appears in that page's flowchart chunk(s)."""
    page = pymupdf.open(PDF_PATH)[pdf_page - 1]
    expected: Counter = Counter()
    for s in _page_spans(page):
        if s["size"] < 20:  # not the chapter heading on p. 29
            expected.update(_numbers(s["text"]))
    charts = [c for c in chunks if c["chunk_type"] == "flowchart" and c["pdf_page_start"] == pdf_page]
    got: Counter = Counter()
    for c in charts:
        got.update(_numbers(verifiable_text(c)))  # a repeated caption only adds counts
    missing = expected - got
    assert not missing, missing


def test_dose_statements_verbatim(chunks: list[dict]) -> None:
    """The document's dose statements (flowcharts and prose) are retrievable verbatim."""
    quotes = {
        "who-hearts-d-2020-fig-1": [
            "IF goal not achieved BEGIN METFORMIN 500 mg daily",
            "INCREASE METFORMIN to 1000 mg 2 x daily",
            "ADD gliclazide 80 mg 2 x daily",
            "GIVE METFORMIN 1000 mg 2 x daily and GLICLAZIDE 80 g 2 x daily",
            "2000 mg metformin and 160 mg gliclazide from diagnosis",
        ],
        "who-hearts-d-2020-fig-2-part1": ["BEGIN METFORMIN 500 mg once daily.", "increase dose to 80 mg 2x daily"],
        "who-hearts-d-2020-fig-2-part2": ["BEGIN gliclazide 80 mg bid"],
        "who-hearts-d-2020-fig-2-part3": [
            "If unconscious, give 20–50 mI of 50% glucose (dextrose) IV over 1–3 minutes.",
            "Set up intravenous drip 0.9% NaCI 1 litre in 2 hours; continue at 1 litre every 4 hours",
            "Fasting | ≤7.0 mmol/l (126mg/dl)†",
        ],
        "who-hearts-d-2020-annex1-flowchart": [
            "START with 10 units intermediate-acting insulin (NPH) at bedtime",
            "INCREASE INSULIN dosage by 1–2 units",
            "REDUCE morning gliclazide by 40 mg",
        ],
        "who-hearts-d-2020-ch3-antiplatelet-treatment": ["75–100 mg of acetylsalicylic acid daily"],
        "who-hearts-d-2020-ch3-acute-complications-of-diabetes-p2": [
            "(20–50 mL of 50% glucose over 1–3 minutes)",
        ],
        "who-hearts-d-2020-ch3-management-of-hyperglycaemic-emergencies": [
            "Infuse isotonic saline (0.9% NaCl) at a rate of 1000 mL in the first 2 hours.",
        ],
    }
    ids = {c["chunk_id"]: c for c in chunks}
    for cid, qs in quotes.items():
        for q in qs:
            assert quote_verifies(q, ids[cid]), (cid, q)


def test_table_rows(by_id: dict[str, dict]) -> None:
    t3 = by_id["who-hearts-d-2020-table-3"]["source_text"].split("\n")
    assert t3[1] == "Measurement | Diagnostic cut-off value | Comment"
    assert "Fasting venous or capillary** plasma glucose | ≥7.0 mmol/L (126 mg/dL) | Least costly but difficulties with ensuring a fasting state" in t3
    assert "HbA1c*** | 6.5% (48 mmol/mol) | • Less intra-individual variability than plasma glucose" in by_id["who-hearts-d-2020-table-3"]["source_text"]
    t1 = by_id["who-hearts-d-2020-table-1"]["source_text"].split("\n")
    # Two sub-rows of one ruled row are kept apart, each with its own description.
    assert any(r.startswith("Monogenic diabetes: a) Monogenic defects of beta-cell function | Caused by") for r in t1)
    assert any(r.startswith("b) Monogenic defects in insulin action | Caused by") for r in t1)
    t8 = by_id["who-hearts-d-2020-table-8"]["source_text"].split("\n")
    assert t8[1] == "Risk level | Low risk | Moderate risk | High risk | Active foot problem"
    assert t8[3] == "Action | Assess annually | Assess every 3–6 months | Assess every 1–3 months | Urgent referral"


# --------------------------------------------------------------------------- #
# Source discrepancies
# --------------------------------------------------------------------------- #


def test_discrepancies(chunks: list[dict], by_id: dict[str, dict]) -> None:
    from src.parse.who_hearts_d import DISCREPANCIES

    flagged = {c["chunk_id"]: c["source_discrepancy"] for c in chunks if c["source_discrepancy"]}
    for d in DISCREPANCIES:
        assert any(d.description in text for text in flagged.values()), d.key
    expected = {
        "who-hearts-d-2020-fig-1": ["'GLICLAZIDE 80 g 2 x daily'", "BMI ≥25", "'≥18 mmol/L'", "'≥2+'"],
        "who-hearts-d-2020-fig-2-part3": ["≤2.8 mmol/L", "'<7.0%'"],
        "who-hearts-d-2020-table-3": ["no '*' marker"],
        "who-hearts-d-2020-table-7": ["'<7.0%'"],
        "who-hearts-d-2020-ch4": ["'>2+'", "'≥18 mmol/L'"],
        "who-hearts-d-2020-annex2-assessing-loss-of-protective-sensation-lops-p1": ["(Fig. 6 and Fig. 7)"],
    }
    for cid, needles in expected.items():
        for needle in needles:
            assert needle in (by_id[cid]["source_discrepancy"] or ""), (cid, needle)
    # The printed values themselves are never corrected.
    assert "GLICLAZIDE 80 g 2 x daily" in by_id["who-hearts-d-2020-fig-1"]["source_text"]
    # Chunks without an anchor carry no discrepancy.
    assert by_id["who-hearts-d-2020-table-1"]["source_discrepancy"] is None


def test_first_review_interval_discrepancy(by_id: dict[str, dict]) -> None:
    for cid in ("who-hearts-d-2020-fig-1", "who-hearts-d-2020-fig-2-part1"):
        assert "Fig. 1 reviews after 1 month" in by_id[cid]["source_discrepancy"], cid
        assert "Fig. 2 reviews after 3 months" in by_id[cid]["source_discrepancy"], cid



# --------------------------------------------------------------------------- #
# Raw file
# --------------------------------------------------------------------------- #


@needs_pdf
def test_raw_file_is_whos_original() -> None:
    """The chunks are built from WHO's original PDF (see data/raw/MANIFEST.md), not a re-print."""
    import hashlib

    assert hashlib.sha256(PDF_PATH.read_bytes()).hexdigest() == (
        "71b216b19a227e73e0e624274974571a3530e30a0db37a3d10b6d179eefac008"
    )
    meta = pymupdf.open(PDF_PATH).metadata
    assert meta["creator"].startswith("Adobe InDesign")


# --------------------------------------------------------------------------- #
# Typo corrections (text only; source_text verbatim)
# --------------------------------------------------------------------------- #

EXPECTED_CORRECTIONS = {
    "who-hearts-d-2020-ch3-acute-complications-of-diabetes-p2": [("to to", "to", 1)],
    "who-hearts-d-2020-fig-2-part3": [("mI", "ml", 1), ("NaCI", "NaCl", 1), ("Iimited", "limited", 1)],
    "who-hearts-d-2020-table-8": [("iscaemia", "ischaemia", 1)],
}


def test_text_corrections_listed(chunks: list[dict]) -> None:
    got = {
        c["chunk_id"]: [(e["printed"], e["corrected"], e["count"]) for e in c["text_corrections"]]
        for c in chunks if c["text_corrections"]
    }
    assert got == EXPECTED_CORRECTIONS
    # Every correction in the map is used by this document.
    assert {p for entries in got.values() for p, _, _ in entries} == {printed for _, printed, _ in TEXT_CORRECTIONS}


def test_corrections_only_in_text(by_id: dict[str, dict]) -> None:
    fig2 = by_id["who-hearts-d-2020-fig-2-part3"]
    assert "20–50 mI of 50% glucose" in fig2["source_text"]  # printed form kept
    assert "0.9% NaCI 1 litre" in fig2["source_text"]
    assert "20–50 ml of 50% glucose" in fig2["text"] and "mI" not in fig2["text"]
    assert "0.9% NaCl 1 litre" in fig2["text"] and "NaCI" not in fig2["text"]
    assert "and/or limited life expectancy" in fig2["text"] and "Iimited" not in fig2["text"]
    hypo = by_id["who-hearts-d-2020-ch3-acute-complications-of-diabetes-p2"]
    assert "unable to to ingest drink" in hypo["source_text"]
    assert "unable to ingest drink" in hypo["text"]
    t8 = by_id["who-hearts-d-2020-table-8"]
    assert "critical limb iscaemia" in t8["source_text"] and "critical limb ischaemia" in t8["text"]


def test_quotes_verify_in_both_spellings(by_id: dict[str, dict]) -> None:
    """A quote verifies whether it uses the printed or the corrected spelling."""
    fig2 = by_id["who-hearts-d-2020-fig-2-part3"]
    for quote in ("give 20–50 mI of 50% glucose", "give 20–50 ml of 50% glucose",
                  "drip 0.9% NaCI 1 litre in 2 hours", "drip 0.9% NaCl 1 litre in 2 hours",
                  "and/or Iimited life expectancy", "and/or limited life expectancy"):
        assert quote_verifies(quote, fig2), quote
    hypo = by_id["who-hearts-d-2020-ch3-acute-complications-of-diabetes-p2"]
    assert quote_verifies("those unable to to ingest drink", hypo)
    assert quote_verifies("those unable to ingest drink", hypo)
    t8 = by_id["who-hearts-d-2020-table-8"]
    assert quote_verifies("critical limb iscaemia", t8) and quote_verifies("critical limb ischaemia", t8)
    # The equivalence is limited to the documented corrections.
    assert not quote_verifies("give 20–50 mg of 50% glucose", fig2)


# --------------------------------------------------------------------------- #
# flow_text (derived)
# --------------------------------------------------------------------------- #

DOSE_RE = re.compile(r"\d+(?:[.–]\d+)?\s*(?:mg|g|mL|ml|mI|units|litre)\b(?!/)")


def test_flow_text_values_come_from_the_figure(chunks: list[dict]) -> None:
    """flow_text adds no value: every number in it is printed in the chunk's source_text;
    and it keeps every dose printed in the figure."""
    for c in [c for c in chunks if c["chunk_type"] == "flowchart"]:
        flow, source = c["flow_text"], c["source_text"]
        assert set(NUMBER_RE.findall(flow)) <= set(NUMBER_RE.findall(source)), c["chunk_id"]
        flow_doses = {normalise_for_match(d) for d in DOSE_RE.findall(flow)}
        for dose in DOSE_RE.findall(source):
            assert normalise_for_match(dose) in flow_doses, (c["chunk_id"], dose)
        for line in flow.split("\n"):
            assert line.count("→") >= 1, (c["chunk_id"], line)


def test_flow_text_is_not_verifiable(chunks: list[dict]) -> None:
    for c in [c for c in chunks if c["chunk_type"] == "flowchart"]:
        assert c["flow_text"] not in verifiable_text(c)
        for line in c["flow_text"].split("\n"):
            assert not quote_verifies(line, c), (c["chunk_id"], line)


def test_fig1_flow_text(by_id: dict[str, dict]) -> None:
    fig1 = by_id["who-hearts-d-2020-fig-1"]
    flow = fig1["flow_text"].split("\n")
    assert "If FPG ≥15 mmol/L and symptomatic, or FPG/RPG >18 mmol/L regardless of symptoms → test urine ketones" in flow
    assert any(line.startswith("At the 1-month review or any of the next four 3-month reviews, if symptomatic and "
                               "FPG ≥15 mmol/L → test urine ketones") for line in flow)
    assert "If urine ketones ≥2+ → refer to higher level of care" in flow
    # The printed value is kept and flagged in flow_text and text; source_text is untouched.
    flagged = "gliclazide 80 g 2 x daily [as printed; 80 mg elsewhere in this document, see source_discrepancy]"
    assert any(flagged in line for line in flow)
    assert flagged in fig1["text"]
    assert "[as printed" not in fig1["source_text"] and "GLICLAZIDE 80 g 2 x daily" in fig1["source_text"]
    assert "gliclazide-unit" not in fig1["text"] and "80 g 2 x daily [as printed" in fig1["text"]


# --------------------------------------------------------------------------- #
# hearts_d_doses.csv
# --------------------------------------------------------------------------- #

DOSES_PATH = CHUNKS_PATH.parents[1] / "interim" / "hearts_d_doses.csv"
DOSE_COLUMNS = ["drug", "drug_as_printed", "drug_source", "dose", "dose_corrected", "dose_basis", "route",
                "frequency", "timing", "dosing_as_printed", "figure_or_section", "step", "branch", "condition",
                "chunk_id", "pdf_page", "printed_page", "source_discrepancy"]
DRUGS = {"metformin", "gliclazide", "insulin", "nph insulin", "glucose", "glucose 50%", "carbohydrate",
         "sodium chloride 0.9%", "acetylsalicylic acid"}
ROUTE_EVIDENCE = {"IV": re.compile(r"\bIV\b|intravenous", re.I), "oral": re.compile(r"\boral\b|\bingest\b", re.I)}


@pytest.fixture(scope="module")
def dose_rows() -> list[dict[str, str]]:
    import csv

    if not DOSES_PATH.exists():
        pytest.fail(f"{DOSES_PATH} missing: run `python -m src.parse.who_hearts_d` first")
    with DOSES_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == DOSE_COLUMNS
        return list(reader)


def test_dose_rows_verbatim(dose_rows: list[dict[str, str]], by_id: dict[str, dict]) -> None:
    """drug_as_printed, dose, frequency, timing, dosing_as_printed and every part of condition
    are verbatim in the row's chunk."""
    assert len(dose_rows) == 29
    for r in dose_rows:
        chunk = by_id[r["chunk_id"]]
        source = normalise_for_match(chunk["source_text"])
        for value in (r["drug_as_printed"], r["dose"], r["frequency"], r["timing"], r["dosing_as_printed"],
                      *[p for p in r["condition"].split("; ") if p]):
            assert normalise_for_match(value) in source, (r["chunk_id"], value)
        for part in (r["frequency"], r["timing"]):
            assert part in r["dosing_as_printed"], r  # split from the verbatim phrase
        assert int(r["pdf_page"]) in range(chunk["pdf_page_start"], chunk["pdf_page_end"] + 1)
        assert r["printed_page"] == r["pdf_page"]


def test_dose_row_drugs(dose_rows: list[dict[str, str]]) -> None:
    for r in dose_rows:
        assert r["drug"] in DRUGS and r["drug"] == r["drug"].lower(), r
        assert r["drug_source"] in {"printed", "inferred_from_sequence"}, r
        assert (r["drug_source"] == "printed") == bool(r["drug_as_printed"]), r
    inferred = [(r["drug"], r["dose"], r["frequency"]) for r in dose_rows if r["drug_source"] == "inferred_from_sequence"]
    assert inferred == [
        ("metformin", "1000 mg", "1x daily"), ("metformin", "1000 mg", "2x daily"), ("gliclazide", "80 mg", "2x daily"),
        ("sodium chloride 0.9%", "1 litre", "every 4 hours"), ("insulin", "1–2 units", "at 3-day intervals"),
        ("sodium chloride 0.9%", "1000 mL", "every 4 hours"),
    ]


def test_dose_row_route_only_when_stated(dose_rows: list[dict[str, str]]) -> None:
    from src.parse.hearts_d_derived import DOSE_STATEMENTS

    assert len(DOSE_STATEMENTS) == len(dose_rows)
    for st, r in zip(DOSE_STATEMENTS, dose_rows):
        assert r["route"] in {"", "oral", "IV"}, r
        if r["route"]:
            stated = " ".join([st.anchor, r["step"], r["drug_as_printed"], r["dosing_as_printed"]])
            assert ROUTE_EVIDENCE[r["route"]].search(stated), r


def test_dose_basis_and_corrections(dose_rows: list[dict[str, str]]) -> None:
    for r in dose_rows:
        assert r["dose_basis"] in {"per_dose", "daily_total_reference", "titration_step", "infusion_rate"}, r
        assert r["dose_corrected"] == apply_corrections(r["dose"])[0], r
    reference = {(r["drug"], r["dose"]) for r in dose_rows if r["dose_basis"] == "daily_total_reference"}
    assert reference == {("metformin", "2000 mg"), ("gliclazide", "160 mg")}
    glucose = [r for r in dose_rows if r["dose"] == "20–50 mI"]
    assert len(glucose) == 1 and glucose[0]["dose_corrected"] == "20–50 ml"
    assert sum(r["dose"] != r["dose_corrected"] for r in dose_rows) == 1


def test_dose_rows_cover_every_flowchart_dose(dose_rows: list[dict[str, str]], chunks: list[dict]) -> None:
    for c in [c for c in chunks if c["chunk_type"] == "flowchart"]:
        doses = Counter(normalise_for_match(d) for d in DOSE_RE.findall(c["source_text"]))
        rows = Counter(normalise_for_match(r["dose"]) for r in dose_rows if r["chunk_id"] == c["chunk_id"])
        assert set(doses) <= set(rows), (c["chunk_id"], set(doses) - set(rows))


def test_dose_rows_keep_printed_values(dose_rows: list[dict[str, str]]) -> None:
    gliclazide_g = [r for r in dose_rows if r["dose"] == "80 g"]
    assert len(gliclazide_g) == 1 and gliclazide_g[0]["drug_as_printed"] == "GLICLAZIDE"
    assert gliclazide_g[0]["dose_corrected"] == "80 g"  # a conflicting value, not a typo: never corrected
    assert "gliclazide-unit" in gliclazide_g[0]["source_discrepancy"]


def test_dose_row_discrepancies_match_chunks(dose_rows: list[dict[str, str]], by_id: dict[str, dict]) -> None:
    from src.parse.who_hearts_d import DISCREPANCIES

    descriptions = {d.key: d.description for d in DISCREPANCIES}
    for r in dose_rows:
        for key in filter(None, r["source_discrepancy"].split("; ")):
            assert descriptions[key] in (by_id[r["chunk_id"]]["source_discrepancy"] or ""), (r, key)
