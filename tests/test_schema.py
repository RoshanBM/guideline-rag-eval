"""The unified chunk schema (src/parse/schema.py), applied to every chunk file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.parse.schema import (
    FIELD_NAMES, build_record, compose_text, quote_verifies, split_text, validate_record, verifiable_text,
)

CHUNK_FILES = sorted((Path(__file__).resolve().parents[1] / "data" / "chunks").glob("*.jsonl"))


def test_chunk_files_exist() -> None:
    assert {p.name for p in CHUNK_FILES} >= {"aware.jsonl", "hypertension.jsonl"}


@pytest.mark.parametrize("path", CHUNK_FILES, ids=lambda p: p.name)
def test_every_file_validates(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records
    errors = {r.get("chunk_id"): validate_record(r) for r in records}
    assert {k: v for k, v in errors.items() if v} == {}
    assert len({r["chunk_id"] for r in records}) == len(records)


def test_chunk_ids_unique_across_collections() -> None:
    ids = [json.loads(line)["chunk_id"] for p in CHUNK_FILES for line in p.read_text(encoding="utf-8").splitlines()]
    assert len(ids) == len(set(ids))


def _record(**overrides: object) -> dict:
    header = "Lower urinary tract infection - adults - antibiotic treatment (equal options):"
    source = "Trimethoprim (oral): 200 mg given every 12 hours | 3 days"
    notes = ["Note. All dosages are for normal renal and hepatic function."]
    aware = "AWaRe category (derived from the colour coding in the source): Trimethoprim – ACCESS"
    values = dict(
        chunk_id="x-1", collection="aware", doc="Doc", doc_id="x", section_path=["Primary health care", "23. X"],
        infection="Lower urinary tract infection", population="adults", population_raw="Adults",
        context_header=header, chunk_type="table", index_default=True, source_text=source,
        aware_category_note=aware, text=compose_text(source, header, notes, aware),
        pdf_page_start=306, pdf_page_end=306, printed_page_start=290, printed_page_end=290,
        source_url="https://example.org", licence="CC BY-NC-SA 3.0 IGO",
    )
    values.update(overrides)
    return build_record(**values)


def test_build_record_orders_fields_and_rejects_unknown() -> None:
    assert list(_record()) == FIELD_NAMES
    with pytest.raises(KeyError):
        build_record(chunk_id="x", not_a_field=1)


def test_text_assembly_round_trips() -> None:
    rec = _record()
    assert validate_record(rec) == []
    header, source, notes, aware = split_text(rec)
    assert rec["text"] == f"{header}\n{source}\nNotes:\n{notes[0]}\n{aware}"
    assert verifiable_text(rec) == f"{source}\n{notes[0]}"


def test_verification_excludes_parser_written_text() -> None:
    rec = _record()
    assert quote_verifies("200 mg given every 12 hours", rec)  # source text
    assert quote_verifies("All dosages are for normal renal and hepatic function", rec)  # footnote
    assert not quote_verifies("adults - antibiotic treatment", rec)  # context header
    assert not quote_verifies("Trimethoprim – ACCESS", rec)  # derived AWaRe category note
    assert not quote_verifies("Notes:", rec)  # section label


def test_validation_catches_problems() -> None:
    assert validate_record(_record(population="infants"))  # not an allowed value
    assert validate_record(_record(population_raw=None))  # population without its raw label
    assert validate_record(_record(text="tampered"))  # text no longer assembled from its parts
    assert validate_record(_record(chunk_type="dosing"))
    bad_order = dict(reversed(list(_record().items())))
    assert validate_record(bad_order)
