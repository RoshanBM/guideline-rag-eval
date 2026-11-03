"""Tests for the eval tooling: eval schema, scripts/validate_eval.py, src/evaluate/relevance.py
and scripts/find_span.py.

The two items below are hand-made examples for these tests only; they are not part of the eval
set (``eval/questions.jsonl``). Their spans are real text from ``data/chunks/hearts_d.jsonl``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pymupdf
import pytest

from src.evaluate.eval_schema import EvalItemError, GoldSource, parse_item, validate_item
from src.evaluate.relevance import load_chunks, relevant_chunk_ids
from src.parse.pdf_utils import PUA_RE
from src.parse.schema import build_record

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import find_span  # noqa: E402
import validate_eval  # noqa: E402

ANSWERABLE = {
    "qid": "example-dose",
    "question": "Example: what daily dose of acetylsalicylic acid does HEARTS-D give for secondary prevention?",
    "category": "dose_lookup",
    "collection": "cardiometabolic",
    "answerable": True,
    "expected_behaviour": "answer",
    "gold_answer": "75–100 mg of acetylsalicylic acid daily.",
    "gold_sources": [
        {"doc_id": "who-hearts-d-2020", "printed_page": 24, "span": "75–100 mg of acetylsalicylic acid daily"},
    ],
    "gold_structured": {"any_of": [
        {"drug": "acetylsalicylic acid", "dose": "75–100 mg", "frequency": "daily", "duration": None},
    ]},
    "notes": "Test fixture only.",
}
UNANSWERABLE = {
    "qid": "example-ood",
    "question": "Example: what is the capital of France?",
    "category": "out_of_domain",
    "collection": None,
    "answerable": False,
    "expected_behaviour": "decline",
    "gold_answer": None,
    "gold_sources": [],
    "gold_structured": None,
    "notes": None,
}
ANTIPLATELET_CHUNK = "who-hearts-d-2020-ch3-antiplatelet-treatment"


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    return load_chunks()


def _write(tmp_path: Path, items: list[object]) -> Path:
    path = tmp_path / "questions.jsonl"
    lines = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _with(item: dict, **changes: object) -> dict:
    out = copy.deepcopy(item)
    out.update(changes)
    return out


def _with_source(item: dict, **changes: object) -> dict:
    out = copy.deepcopy(item)
    out["gold_sources"][0].update(changes)
    return out


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_examples_are_valid_items() -> None:
    assert validate_item(ANSWERABLE) == []
    assert validate_item(UNANSWERABLE) == []
    item = parse_item(ANSWERABLE)
    assert item.gold_sources[0].printed_page == 24
    assert item.gold_structured is not None and item.gold_structured.any_of[0].dose == "75–100 mg"


@pytest.mark.parametrize("bad, message", [
    (_with(UNANSWERABLE, gold_sources=ANSWERABLE["gold_sources"]), "unanswerable items must have empty gold_sources"),
    (_with(UNANSWERABLE, expected_behaviour="answer"), "expected_behaviour 'decline' or 'redirect'"),
    (_with(ANSWERABLE, gold_sources=[]), "at least one gold source"),
    (_with(ANSWERABLE, expected_behaviour="decline"), "expected_behaviour 'answer'"),
    (_with(ANSWERABLE, collection=None), "need a collection"),
    (_with(ANSWERABLE, category="dosing"), "category 'dosing' not in"),
    (_with_source(ANSWERABLE, printed_page="24"), "printed_page must be an integer"),
    (_with(ANSWERABLE, gold_structured={"any_of": []}), "any_of must be a non-empty list"),
    (_with(ANSWERABLE, gold_structured={"any_of": [{"drug": "x", "dose": "1 mg"}]}), "missing keys"),
    ({k: v for k, v in ANSWERABLE.items() if k != "notes"}, "missing keys ['notes']"),
    (_with(ANSWERABLE, extra=1), "unknown keys ['extra']"),
])
def test_schema_rejects(bad: dict, message: str) -> None:
    errors = validate_item(bad)
    assert any(message in e for e in errors), errors
    with pytest.raises(EvalItemError):
        parse_item(bad)


# --------------------------------------------------------------------------- #
# scripts/validate_eval.py
# --------------------------------------------------------------------------- #


def test_validate_examples(tmp_path: Path, chunks: list[dict]) -> None:
    report = validate_eval.validate_file(_write(tmp_path, [ANSWERABLE, UNANSWERABLE]), chunks)
    assert report.errors == [] and report.warnings == []
    assert report.counts == {"dose_lookup": 1, "out_of_domain": 1}


def test_validate_cli_counts_and_strict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, [ANSWERABLE, UNANSWERABLE])
    assert validate_eval.main(["--questions", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 valid item(s), 0 error(s)" in out
    assert "dose_lookup              1      12  (11 to go)" in out
    assert "total                    2      59  (+1 spare allowed)" in out
    assert validate_eval.main(["--questions", str(path), "--strict"]) == 1  # far below the targets


@pytest.mark.parametrize("item, message", [
    # the span is real but on printed page 24, not 23
    (_with_source(ANSWERABLE, printed_page=23), "span not found in who-hearts-d-2020 on printed page 23; "
                                                "it occurs in who-hearts-d-2020 p. 24"),
    # not verbatim: matching keeps case
    (_with_source(ANSWERABLE, span="75–100 MG of acetylsalicylic acid daily"), "it occurs in no chunk"),
    (_with_source(ANSWERABLE, doc_id="who-hearts-d-2023"), "unknown doc_id 'who-hearts-d-2023'"),
    (_with(ANSWERABLE, collection="aware"), "belongs to collection 'cardiometabolic', not 'aware'"),
    (_with(ANSWERABLE, collection="oncology"), "unknown collection 'oncology'"),
    (_with(UNANSWERABLE, gold_sources=ANSWERABLE["gold_sources"]), "unanswerable items must have empty gold_sources"),
])
def test_validate_reports_errors(tmp_path: Path, chunks: list[dict], item: dict, message: str) -> None:
    report = validate_eval.validate_file(_write(tmp_path, [item]), chunks)
    assert any(message in e for e in report.errors), report.errors


def test_validate_duplicates_and_bad_json(tmp_path: Path, chunks: list[dict]) -> None:
    report = validate_eval.validate_file(_write(tmp_path, [ANSWERABLE, ANSWERABLE, "{not json"]), chunks)
    assert any("line 2 (example-dose): duplicate qid" in e for e in report.errors), report.errors
    assert any(e.startswith("line 3: invalid JSON") for e in report.errors), report.errors


def test_validate_span_normalisation(tmp_path: Path, chunks: list[dict]) -> None:
    """Whitespace, dash style and the documented typo corrections do not matter; the printed and
    the corrected spelling of 'mI' both verify against Fig. 2 (printed page 25)."""
    items = [
        _with_source(ANSWERABLE, span="75-100 mg of\nacetylsalicylic   acid daily"),
        _with_source(_with(ANSWERABLE, qid="example-printed"), printed_page=25,
                     span="If unconscious, give 20–50 mI of 50% glucose (dextrose) IV over 1–3 minutes."),
        _with_source(_with(ANSWERABLE, qid="example-corrected"), printed_page=25,
                     span="If unconscious, give 20–50 ml of 50% glucose (dextrose) IV over 1–3 minutes."),
    ]
    report = validate_eval.validate_file(_write(tmp_path, items), chunks)
    assert report.errors == []


def test_validate_flags_short_spans(tmp_path: Path, chunks: list[dict]) -> None:
    report = validate_eval.validate_file(_write(tmp_path, [_with_source(ANSWERABLE, span="75–100 mg")]), chunks)
    assert report.errors == [] and any("fewer than 3 words" in w for w in report.warnings)


def test_validate_missing_file(tmp_path: Path, chunks: list[dict]) -> None:
    report = validate_eval.validate_file(tmp_path / "absent.jsonl", chunks)
    assert report.errors and "does not exist" in report.errors[0]


# --------------------------------------------------------------------------- #
# src/evaluate/relevance.py
# --------------------------------------------------------------------------- #


def test_relevant_chunk_ids(chunks: list[dict]) -> None:
    assert relevant_chunk_ids(ANSWERABLE, chunks) == [ANTIPLATELET_CHUNK]
    assert relevant_chunk_ids(parse_item(ANSWERABLE), chunks) == [ANTIPLATELET_CHUNK]  # EvalItem works too
    assert relevant_chunk_ids(UNANSWERABLE, chunks) == []


def test_relevance_requires_document_and_page(chunks: list[dict]) -> None:
    assert relevant_chunk_ids(_with_source(ANSWERABLE, printed_page=23), chunks) == []
    assert relevant_chunk_ids(_with_source(ANSWERABLE, doc_id="who-htn-2021"), chunks) == []


def test_relevance_union_of_sources_and_footnotes(chunks: list[dict]) -> None:
    """Several gold sources give the union of their chunks, in chunk order; a span in a footnote
    (here Table 3's note) counts, because footnotes are verbatim source text."""
    item = copy.deepcopy(ANSWERABLE)
    item["gold_sources"].insert(0, {"doc_id": "who-hearts-d-2020", "printed_page": 13,
                                    "span": "Overnight fast of 8–14 hours."})
    assert relevant_chunk_ids(item, chunks) == ["who-hearts-d-2020-table-3", ANTIPLATELET_CHUNK]


def test_relevance_survives_rechunking(chunks: list[dict]) -> None:
    """Relevance depends on spans, not chunk ids: re-chunk HEARTS-D one line per chunk and the
    new chunk holding the span is found."""
    rechunked = []
    for c in chunks:
        if c["doc_id"] != "who-hearts-d-2020":
            continue
        for i, line in enumerate(c["source_text"].split("\n")):
            rechunked.append(build_record(
                chunk_id=f"{c['chunk_id']}-line{i}", collection=c["collection"], doc=c["doc"], doc_id=c["doc_id"],
                section_path=c["section_path"], chunk_type="prose", index_default=True, source_text=line, text=line,
                pdf_page_start=c["pdf_page_start"], pdf_page_end=c["pdf_page_end"],
                printed_page_start=c["printed_page_start"], printed_page_end=c["printed_page_end"],
                source_url=c["source_url"], licence=c["licence"],
            ))
    ids = relevant_chunk_ids(ANSWERABLE, rechunked)
    assert len(ids) == 1 and ids[0].startswith(f"{ANTIPLATELET_CHUNK}-line")


# --------------------------------------------------------------------------- #
# scripts/find_span.py
# --------------------------------------------------------------------------- #


def test_find_span_phrase(chunks: list[dict]) -> None:
    results = find_span.find("ACETYLSALICYLIC acid daily", chunks)  # case-insensitive
    assert [(r["chunk_id"], r["match"]) for r in results] == [(ANTIPLATELET_CHUNK, "phrase")]
    span = results[0]["span"]
    assert span.startswith("The recommendation is for 75–100 mg of acetylsalicylic acid daily")
    # The printed span is ready to paste: it validates as a gold source.
    assert relevant_chunk_ids(_with_source(ANSWERABLE, span=span), chunks) == [ANTIPLATELET_CHUNK]


def test_find_span_corrections_and_all_words(chunks: list[dict]) -> None:
    corrected = find_span.find("20–50 ml of 50% glucose (dextrose)", chunks)
    assert [r["chunk_id"] for r in corrected] == ["who-hearts-d-2020-fig-2-part3"]
    assert "20–50 mI" in corrected[0]["span"]  # printed spelling returned
    words = find_span.find("cefalexin 375 mg", chunks, doc_id="who-aware-2022")
    assert words and all(r["match"] == "all words" for r in words)
    assert all("Cefalexin" in r["span"] and "375 mg" in r["span"] for r in words)
    assert find_span.find("no such drug 999 mg", chunks) == []


# --------------------------------------------------------------------------- #
# Only verbatim source text counts
# --------------------------------------------------------------------------- #

FIG1 = "who-hearts-d-2020-fig-1"


def _by_id(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


def _parser_written_spans(chunks: list[dict]) -> list[tuple[str, str, int]]:
    """(field, span, printed page) for text that exists only in parser-written fields."""
    by_id = _by_id(chunks)
    fig1, antiplatelet = by_id[FIG1], by_id[ANTIPLATELET_CHUNK]
    aware = next(c for c in chunks if c["aware_category_note"])
    return [
        ("flow_text", "At the 1-month review, if goal not achieved → begin metformin 500 mg daily", 16),
        ("context_header", antiplatelet["context_header"], 24),
        ("aware_category_note", aware["aware_category_note"], aware["printed_page_start"]),
    ]


def test_parser_written_text_never_verifies(tmp_path: Path, chunks: list[dict]) -> None:
    """A span present only in flow_text, context_header or aware_category_note is rejected by the
    validator and has no relevant chunk."""
    by_id = _by_id(chunks)
    assert "At the 1-month review, if goal not achieved → begin metformin 500 mg daily" in by_id[FIG1]["flow_text"]
    for field_name, span, page in _parser_written_spans(chunks):
        doc_id = "who-aware-2022" if field_name == "aware_category_note" else "who-hearts-d-2020"
        item = _with_source(_with(ANSWERABLE, collection="aware" if doc_id == "who-aware-2022" else "cardiometabolic"),
                            doc_id=doc_id, printed_page=page, span=span)
        assert relevant_chunk_ids(item, chunks) == [], field_name
        report = validate_eval.validate_file(_write(tmp_path, [item]), chunks, page_check=False)
        assert any("it occurs in no chunk" in e for e in report.errors), (field_name, report.errors)


def test_flow_text_only_span_is_rejected(tmp_path: Path, chunks: list[dict]) -> None:
    """The derived flow_text wording is rejected even though the same dose is printed in Fig. 1."""
    span = "If urine ketones <2+ → give metformin 1000 mg 2 x daily"
    assert span in _by_id(chunks)[FIG1]["flow_text"]
    item = _with_source(ANSWERABLE, printed_page=16, span=span)
    report = validate_eval.validate_file(_write(tmp_path, [item]), chunks)
    assert report.errors and "it occurs in no chunk" in report.errors[0]
    assert relevant_chunk_ids(item, chunks) == []
    printed = _with_source(ANSWERABLE, printed_page=16, span="GIVE METFORMIN 1000 mg 2 x daily")
    assert relevant_chunk_ids(printed, chunks) == [FIG1]


# --------------------------------------------------------------------------- #
# Warnings
# --------------------------------------------------------------------------- #


def test_warns_when_span_matches_several_chunks(tmp_path: Path, chunks: list[dict]) -> None:
    """AWaRe repeats a table caption in the four dosing chunks of Table 5.3 (printed p. 44)."""
    item = _with_source(_with(ANSWERABLE, collection="aware"), doc_id="who-aware-2022", printed_page=44,
                        span="Table 5.3 – Empiric antibiotic treatment for acute bacterial otitis media")
    report = validate_eval.validate_file(_write(tmp_path, [item]), chunks)
    assert report.errors == []
    assert len(report.warnings) == 1 and "span matches 4 chunks" in report.warnings[0]
    assert "who-aware-2022-ch05-dosing-adults-first-choice" in report.warnings[0]


# --------------------------------------------------------------------------- #
# Page check against the raw PDF (validate_eval.py only)
# --------------------------------------------------------------------------- #

needs_pdfs = pytest.mark.skipif(
    not all(p.exists() for p in (ROOT / "data" / "raw" / f for f in (
        "WHO-UCN-NCD-20.1-eng.pdf", "Bookshelf_NBK573631.pdf", "9789240062382-eng.pdf"))),
    reason="raw PDFs not present",
)


@needs_pdfs
@pytest.mark.parametrize("doc_id, page, span", [
    # HEARTS-D: superscript footnote marker "1" between "insulin" and "to" in the raw text
    ("who-hearts-d-2020", 15, "Refer for insulin treatment or add human insulin to oral medication (Annex 1)."),
    # HEARTS-D: a sentence running over the page break is cited at the page where it starts
    ("who-hearts-d-2020", 9, "The majority of people with T2DM are overweight or obese, which either causes or "
                             "aggravates insulin resistance."),
    # HEARTS-D: printed typo and its correction both pass ("Iimited" is in the raw text)
    ("who-hearts-d-2020", 25, "and/or Iimited life expectancy."),
    ("who-hearts-d-2020", 25, "and/or limited life expectancy."),
    # AWaRe (chunks from the EPUB, check against the PDF): footnote letter "d" glued in the raw text
    ("who-aware-2022", 59, "Clarithromycin (oral): 500 mg given every 12 hours"),
    # Hypertension (browser print): "benefits" is drawn with a Private Use Area ligature glyph
    ("who-htn-2021", 7, "The anticipated benefits of a lower blood pressure (BP) target"),
])
def test_page_check_accepts(doc_id: str, page: int, span: str) -> None:
    assert validate_eval.PageChecker().check(GoldSource(doc_id, page, span)) is None


@needs_pdfs
def test_hypertension_page_uses_glyph_table() -> None:
    """The hypertension page really contains Private Use Area glyphs, so the check above needs
    the parser's glyph table."""
    page = pymupdf.open(ROOT / "data" / "raw" / "Bookshelf_NBK573631.pdf")[18]  # printed p. 7
    assert PUA_RE.search(page.get_text())


@needs_pdfs
@pytest.mark.parametrize("doc_id, page, span, real_page", [
    # the chunk covers printed pp. 18-19, so the chunk check passes, but the sentence is on p. 18
    ("who-hearts-d-2020", 19, "Two important acute complications are hypoglycaemia and hyperglycaemic emergencies.", 18),
    # the AWaRe chunk covers pp. 58-59; the row is printed on p. 59
    ("who-aware-2022", 58, "Clarithromycin (oral): 500 mg given every 12 hours", 59),
])
def test_page_check_rejects_wrong_page(tmp_path: Path, chunks: list[dict], doc_id: str, page: int, span: str,
                                       real_page: int) -> None:
    collection = "aware" if doc_id == "who-aware-2022" else "cardiometabolic"
    item = _with_source(_with(ANSWERABLE, collection=collection), doc_id=doc_id, printed_page=page, span=span)
    assert relevant_chunk_ids(item, chunks)  # relevance stays at chunk level
    report = validate_eval.validate_file(_write(tmp_path, [item]), chunks)
    assert len(report.errors) == 1, report.errors
    assert f"span not in the raw text of printed page {page}" in report.errors[0]
    assert f"it is on printed page(s) [{real_page}]" in report.errors[0]
    # The page check is the only difference: skipping it, the item is valid.
    assert validate_eval.validate_file(_write(tmp_path, [item]), chunks, page_check=False).errors == []


@needs_pdfs
def test_page_crossing_span_cited_at_second_page_is_rejected() -> None:
    span = ("The majority of people with T2DM are overweight or obese, which either causes or aggravates "
            "insulin resistance.")
    assert "printed page 10" in (validate_eval.PageChecker().check(GoldSource("who-hearts-d-2020", 10, span)) or "")


def test_page_check_without_raw_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> None:
        raise FileNotFoundError("data/raw/absent.pdf")

    monkeypatch.setattr(validate_eval, "get_source", missing)
    problem = validate_eval.PageChecker().check(GoldSource(**ANSWERABLE["gold_sources"][0]))
    assert problem is not None and "use --skip-page-check" in problem


def test_page_key() -> None:
    """Spaces, punctuation and line-end hyphenation are ignored; numbers keep their decimal point
    and range dash; comparison signs are kept."""
    key = validate_eval.page_key
    assert key("signi-\nficant 7.5 mg") == key("significant 7.5 mg")
    assert key("7.5 mg") != key("75 mg")
    assert key("3–6 kg") == key("3-6 kg") != key("36 kg")
    assert key("≥18 mmol/L") != key("<18 mmol/L")
    assert key("1.73 m²") == key("1.73 m")  # the raw superscript is dropped, so the exponent is too
    assert key("20–50 mI") == key("20-50 ml")
