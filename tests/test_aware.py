"""Tests for the AWaRe primary health care outputs.

Regenerate the files with ``python -m src.parse.who_aware`` before running.
Tests that need the raw EPUB/PDF are skipped when they are not present.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from src.parse.schema import quote_verifies, split_text, validate_record
from src.parse.sources import SOURCES
from src.parse.text_utils import normalise_text

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "chunks" / "aware.jsonl"
DOSING_PATH = ROOT / "data" / "interim" / "aware_dosing.csv"
NO_ABX_PATH = ROOT / "data" / "interim" / "aware_no_antibiotic.csv"
RAW_AVAILABLE = SOURCES["aware_epub"].path.exists() and SOURCES["aware_pdf"].path.exists()

CONSTANTS = {
    "collection": "aware",
    "doc": "The WHO AWaRe (Access, Watch, Reserve) antibiotic book (2022)",
    "doc_id": "who-aware-2022",
    "source_url": "https://iris.who.int/handle/10665/365237",
    "licence": "CC BY-NC-SA 3.0 IGO",
    "rec_id": None, "rec_part": None, "parent_rec_id": None, "index_default": True,
    "strength": None, "strength_as_printed": None, "certainty": None,
}
DOSING_COLUMNS = [
    "chapter_no", "infection", "population", "population_raw", "line", "antibiotic", "aware_category", "dose",
    "frequency", "dose_basis", "resolved_dose", "resolved_frequency", "resolved_rule", "resolved_from",
    "route", "duration", "weight_band", "dose_unit_note", "notes", "chunk_id", "pdf_page", "printed_page",
]

# Rows where no frequency is printed and the duration states no schedule (resolved_rule
# "not_stated"), keyed by (chapter, antibiotic, dose).
NOT_STATED_FREQUENCY = {
    ("21", "Ceftriaxone", "1 g"),  # syphilis in pregnancy, allergy alternative: "Ceftriaxone 1 g for 10–14 days"
    ("22", "Metronidazole", "2 g"),  # Table 22.3: durations not assignable to the two regimens
}

# "use adult dose" rows that cannot be linked to a unique adult row (several adult options that
# the route does not disambiguate). Keyed by (chapter, antibiotic, weight band). All 12 such rows
# in the primary care chapters currently resolve, so this is empty; add a row here, with the
# reason, only if the source genuinely leaves the adult dose ambiguous.
UNRESOLVED_ADULT_DOSE: set[tuple[str, str, str]] = set()

# Reviewed (population_raw -> population) mappings. population_raw is the verbatim label the
# value was derived from: a column header, an in-cell heading, an in-cell sub-group, or a
# phrase in a row label / caption.
POPULATION_MAP = {
    ("Adults", "adults"), ("adults", "adults"),
    ("adults and adolescents", "adults_adolescents"), ("adults, adolescents", "adults_adolescents"),
    ("Children", "children"), ("children", "children"), ("Infants", "children"),
    ("Neonates and children", "children"),  # neonates are a subset of children
    ("Neonates", "neonates"), ("First week of life", "neonates"),
    ("Ophthalmia neonatorum", "neonates"), ("ophthalmia neonatorum", "neonates"),
    ("Adults and children", "all"), ("Children and adults", "all"), ("children and adults", "all"),
    ("Infection in pregnant women", "pregnant"), ("Syphilis in pregnancy", "pregnant"),
}
IN_SCOPE_CHAPTERS = set(range(4, 24))  # 4. Bronchitis ... 23. Lower urinary tract infection
MAX_WORDS = 600

# Chapters in scope that genuinely have no antibiotic dosing.
NO_DOSING_CHAPTERS = {
    4: "Bronchitis: antibiotic treatment is not recommended; the only medicine table is Table 4.2 "
       "(symptomatic treatment with ibuprofen/paracetamol), which is not an antibiotic.",
}

# Dosing rows with an empty field because the source does not print it or is ambiguous
# (see the parser summary). Keyed by (chapter, antibiotic, dose) -> missing fields.
KNOWN_INCOMPLETE_ROWS = {
    # Syphilis in pregnancy, penicillin-allergy alternatives written as free text without a route
    # and outside the coloured AWaRe spans (Table 21.5, printed p. 269).
    ("21", "Ceftriaxone", "1 g"): {"route", "aware_category", "frequency"},
    ("21", "Azithromycin", "2 g"): {"route", "aware_category"},
    ("21", "erythromycin", "500 mg"): {"route", "aware_category"},
    ("21", "Erythromycin", "500 mg"): {"route", "aware_category"},
    # Table 22.3 lists two metronidazole regimens and two durations ("Single dose", "7 days")
    # without saying which goes with which.
    ("22", "Metronidazole", "2 g"): {"frequency", "duration"},
    ("22", "Metronidazole", "400 or 500 mg"): {"duration"},
}

# Inconsistencies in the source, recorded in source_discrepancy (not silently resolved).
EXPECTED_DISCREPANCIES = {
    "who-aware-2022-ch05-dosing-children-first-choice",  # amoxicillin band "3–6 kg" vs "3–< 6 kg"
    "who-aware-2022-ch06-dosing-children-first-choice",  # same
    "who-aware-2022-ch18-table-18-2",  # caption missing from the EPUB, taken from the PDF
}


@pytest.fixture(scope="module")
def chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        pytest.fail(f"{CHUNKS_PATH} missing: run `python -m src.parse.who_aware` first")
    return [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def by_id(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


@pytest.fixture(scope="module")
def dosing() -> list[dict[str, str]]:
    with DOSING_PATH.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == DOSING_COLUMNS
        return list(reader)


@pytest.fixture(scope="module")
def no_abx() -> list[dict[str, str]]:
    with NO_ABX_PATH.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def chapter_of(chunk: dict) -> int:
    return int(re.match(r"who-aware-2022-ch(\d\d)-", chunk["chunk_id"]).group(1))


def is_dosing_chunk(chunk: dict) -> bool:
    return "-dosing-" in chunk["chunk_id"] or chunk["chunk_id"].endswith("-dosing")


# --------------------------------------------------------------------------- #
# Chunk schema
# --------------------------------------------------------------------------- #
def test_every_chunk_matches_schema(chunks: list[dict]) -> None:
    for c in chunks:
        cid = c.get("chunk_id")
        assert validate_record(c) == [], cid  # unified schema, src/parse/schema.py
        for key, value in CONSTANTS.items():
            assert c[key] == value, (cid, key)
        assert re.fullmatch(r"who-aware-2022-ch\d\d-[a-z0-9]+(-[a-z0-9]+)*", cid), cid
        assert c["chunk_type"] in {"prose", "table", "no_antibiotic"}, cid
        path = c["section_path"]
        assert path[0] == "Primary health care" and re.match(r"^\d+\. ", path[1]), cid
        assert c["infection"] and path[1].endswith(c["infection"]), cid
        for key in ("printed_page_start", "printed_page_end"):
            assert isinstance(c[key], int), (cid, key)
        assert c["printed_page_start"] <= c["printed_page_end"], cid


def test_population_labels(chunks: list[dict], dosing: list[dict[str, str]]) -> None:
    """Every (population_raw -> population) pair is one of the reviewed mappings."""
    for c in chunks:
        if c["population"] is not None:
            assert (c["population_raw"], c["population"]) in POPULATION_MAP, c["chunk_id"]
    for i, r in enumerate(dosing, start=2):
        assert bool(r["population"]) == bool(r["population_raw"]), i
        if r["population"]:
            assert (r["population_raw"], r["population"]) in POPULATION_MAP, (i, r["population_raw"])
    # The raw label is verbatim source text: it appears in the chunk's section path or source text.
    for c in chunks:
        if c["population_raw"]:
            where = c["source_text"] + "\n" + "\n".join(c["section_path"])
            assert c["population_raw"] in where, c["chunk_id"]


def test_chunk_ids_unique(chunks: list[dict]) -> None:
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_no_chunk_empty_or_too_long(chunks: list[dict]) -> None:
    for c in chunks:
        n = len(c["text"].split())
        assert 0 < n <= MAX_WORDS, (c["chunk_id"], n)


def test_only_primary_care_chapters(chunks: list[dict]) -> None:
    assert {chapter_of(c) for c in chunks} == IN_SCOPE_CHAPTERS


def test_pdf_pages_within_pdf(chunks: list[dict]) -> None:
    if not SOURCES["aware_pdf"].path.exists():
        pytest.skip("raw PDF not available")
    import pymupdf

    doc = pymupdf.open(SOURCES["aware_pdf"].path)
    for c in chunks:
        assert 1 <= c["pdf_page_start"] <= c["pdf_page_end"] <= doc.page_count, c["chunk_id"]
        # The printed page is the PDF's own page label for that page (no offset assumed).
        assert doc[c["pdf_page_start"] - 1].get_label() == str(c["printed_page_start"]), c["chunk_id"]


def test_text_normalised(chunks: list[dict]) -> None:
    for c in chunks:
        assert normalise_text(c["text"]) == c["text"], c["chunk_id"]
        assert not re.search("[-]", c["text"]), c["chunk_id"]


# --------------------------------------------------------------------------- #
# Retrieval text vs source text
# --------------------------------------------------------------------------- #
def test_context_headers(chunks: list[dict]) -> None:
    for c in chunks:
        header = c["context_header"]
        needs_header = c["chunk_type"] == "no_antibiotic" or is_dosing_chunk(c)
        if needs_header:
            assert header, c["chunk_id"]
        if header is None:
            continue
        assert "\n" not in header and header.endswith(":"), c["chunk_id"]
        assert header.startswith(c["infection"]), c["chunk_id"]
        assert c["text"].startswith(header + "\n"), c["chunk_id"]
        assert header not in c["source_text"], c["chunk_id"]


def test_aware_category_note(chunks: list[dict], dosing: list[dict[str, str]]) -> None:
    """A dosing chunk has a category note exactly when its antibiotics are colour-coded in the source."""
    coded = defaultdict(set)
    for r in dosing:
        if r["aware_category"]:
            coded[r["chunk_id"]].add((r["antibiotic"], r["aware_category"].upper()))
    for c in chunks:
        note = c["aware_category_note"]
        if c["chunk_id"] in coded:
            assert note and note.startswith("AWaRe category (derived from the colour coding in the source): "), c["chunk_id"]
            for name, cat in coded[c["chunk_id"]]:
                assert f"{name} – {cat}" in note, (c["chunk_id"], name)
        else:
            # Non-dosing chunks, and the syphilis-in-pregnancy allergy alternatives (not colour-coded).
            assert note is None, c["chunk_id"]
        # The derived note never leaks into the source text.
        assert "AWaRe category" not in c["source_text"], c["chunk_id"]


def test_footnotes_under_notes(chunks: list[dict]) -> None:
    for c in chunks:
        _, source, notes, _ = split_text(c)
        if notes:
            assert c["chunk_type"] == "table", c["chunk_id"]
            assert "\nNotes:\n" in c["text"], c["chunk_id"]
        # Footnotes are not repeated inside the source text, and the colour legend is never kept.
        for note in notes:
            assert note not in source, (c["chunk_id"], note)
            assert not note.startswith("Legend:"), c["chunk_id"]
    # Table 23.5 (printed pp. 290-291): the adults chunk uses footnotes a, b and d plus the general note;
    # footnote c (children's oral liquid formulation) is referenced only from the children column.
    by = {c["chunk_id"]: c for c in chunks}
    adult_notes = split_text(by["who-aware-2022-ch23-dosing-adults"])[2]
    assert [n[:2] for n in adult_notes] == ["No", "a ", "b ", "d "]
    assert adult_notes[0] == "Note. All dosages are for normal renal and hepatic function."
    child_notes = split_text(by["who-aware-2022-ch23-dosing-children"])[2]
    assert any(n.startswith("c Oral liquid formulations must be refrigerated") for n in child_notes)


def test_quotes_verify_against_source_text_only(chunks: list[dict]) -> None:
    """Quotes verify against source text and footnotes, never the header or the category note."""
    for c in chunks:
        source = c["source_text"]
        assert quote_verifies(source, c), c["chunk_id"]
        for start in range(0, len(source), max(1, len(source) // 4)):
            quote = source[start : start + 70]
            if quote.strip():
                assert quote_verifies(quote, c), (c["chunk_id"], quote)
        for note in split_text(c)[2]:
            assert quote_verifies(note, c), (c["chunk_id"], note)
        if c["context_header"]:
            assert not quote_verifies(c["context_header"], c), c["chunk_id"]
        if c["aware_category_note"]:
            assert not quote_verifies(c["aware_category_note"], c), c["chunk_id"]


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_every_chapter_has_table_chunk_and_dosing_rows(chunks: list[dict], dosing: list[dict[str, str]]) -> None:
    tables = defaultdict(int)
    for c in chunks:
        if c["chunk_type"] == "table":
            tables[chapter_of(c)] += 1
    rows = defaultdict(int)
    for r in dosing:
        rows[int(r["chapter_no"])] += 1
    for ch in sorted(IN_SCOPE_CHAPTERS):
        assert tables[ch] >= 1, f"chapter {ch} has no table chunk"
        if ch in NO_DOSING_CHAPTERS:
            assert rows[ch] == 0, f"chapter {ch} listed as having no dosing but has rows"
        else:
            assert rows[ch] >= 1, f"chapter {ch} has no dosing rows"


def test_every_dosing_table_chunk_has_rows(chunks: list[dict], dosing: list[dict[str, str]]) -> None:
    used = {r["chunk_id"] for r in dosing}
    for c in chunks:
        if is_dosing_chunk(c):
            assert c["chunk_id"] in used, c["chunk_id"]


# --------------------------------------------------------------------------- #
# Dosing CSV
# --------------------------------------------------------------------------- #
def test_dosing_rows_point_to_chunks_containing_their_values(dosing: list[dict[str, str]], by_id: dict[str, dict]) -> None:
    for i, r in enumerate(dosing, start=2):
        chunk = by_id.get(r["chunk_id"])
        assert chunk is not None, (i, r["chunk_id"])
        assert chunk["chunk_type"] == "table" and is_dosing_chunk(chunk), (i, r["chunk_id"])
        # Verified against source text + footnotes only: the derived AWaRe note also names every
        # antibiotic, so checking "text" would let a name through that the source text lacks.
        assert quote_verifies(r["antibiotic"], chunk), (i, r["antibiotic"])
        if r["dose"]:
            assert quote_verifies(r["dose"], chunk), (i, r["dose"])
        if r["weight_band"]:
            assert quote_verifies(r["weight_band"], chunk), (i, r["weight_band"])
        assert chunk["pdf_page_start"] <= int(r["pdf_page"]) <= chunk["pdf_page_end"], i
        assert chunk["printed_page_start"] <= int(r["printed_page"]) <= chunk["printed_page_end"], i


def test_dosing_values_are_valid(dosing: list[dict[str, str]]) -> None:
    for i, r in enumerate(dosing, start=2):
        assert r["line"] in {"first_choice", "second_choice", "alternative", "allergy", "other"}, i
        assert r["aware_category"] in {"Access", "Watch", "Reserve", ""}, i
        assert r["population"] in {"adults", "adults_adolescents", "children", "neonates", "pregnant", "all", ""}, i
        assert r["antibiotic"], i
        assert r["weight_band"] == "" or "kg" in r["weight_band"], i


def test_duration_footnotes_reach_rows(dosing: list[dict[str, str]], by_id: dict[str, dict]) -> None:
    """Every footnote marker printed in a duration cell or on a duration column header appears,
    as "Footnote <letter> <text>", in the notes of every CSV row that uses that duration."""
    if not RAW_AVAILABLE:
        pytest.skip("raw EPUB not available")
    from src.parse.who_aware import cell_paras, column_role, footnote_marks, load_chapters, table_grid

    rows_by_caption: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in dosing:
        rows_by_caption[by_id[r["chunk_id"]]["section_path"][-1]].append(r)
    found: set[tuple[str, str, str]] = set()
    for ch in load_chapters(SOURCES["aware_epub"].path):
        for table in ch.soup.find_all("table"):
            if not table.find("span", class_=re.compile(r"^back_")):
                continue
            grid = table_grid(table, ch.number)
            table_no = grid.caption.split(" – ")[0]
            rows = rows_by_caption[grid.caption]
            foot = {letter: text for letter, text, _ in grid.foot if letter}
            for k, header in enumerate(grid.header):
                if column_role(header)[0] != "duration":
                    continue
                # Header marker: applies to every duration taken from this column (allergy
                # alternatives state their own durations in free text).
                header_el = grid.header_els[k]
                for mark in footnote_marks(header_el) if header_el is not None else set():
                    found.add((table_no, "header", mark))
                    affected = [r for r in rows if r["duration"] and r["line"] != "allergy"]
                    assert affected, (table_no, mark)
                    for r in affected:
                        assert f"Footnote {foot[mark]}" in r["notes"], (table_no, mark, r["antibiotic"], r["weight_band"])
                # Cell markers: apply to the rows whose duration is (or includes) that paragraph.
                seen: set[int] = set()
                for row in grid.rows:
                    cell = row[k]
                    if cell is None or id(cell) in seen:
                        continue
                    seen.add(id(cell))
                    for para in cell_paras(cell):
                        if not para.marks:
                            continue
                        affected = [r for r in rows if r["duration"] and para.text in r["duration"]]
                        assert affected, (table_no, para.text)
                        for mark in para.marks:
                            found.add((table_no, "cell", mark))
                            for r in affected:
                                assert f"Footnote {foot[mark]}" in r["notes"], (table_no, mark, r["antibiotic"], r["weight_band"])
    # The markers printed in the book (EPUB), so the test cannot pass by checking nothing.
    assert found == {
        ("Table 6.5", "cell", "b"), ("Table 6.5", "cell", "c"), ("Table 8.4", "cell", "c"),
        ("Table 16.4", "cell", "d"), ("Table 23.5", "header", "d"),
    }


def test_dose_unit_notes(dosing: list[dict[str, str]]) -> None:
    """Weight-band sub-headings are in dose_unit_note, not in notes."""
    for i, r in enumerate(dosing, start=2):
        assert "weight bands" not in r["notes"] and "sulfamethoxazole/trimethoprim component" not in r["notes"], i
        if r["dose_unit_note"]:
            assert r["weight_band"], i
    assert {r["dose_unit_note"] for r in dosing} == {
        "", "Oral weight bands",
        "Oral weight bands (mg of the sulfamethoxazole/trimethoprim component)",  # Table 14.6
        "Oral weight bands; mg of sulfamethoxazole/trimethoprim component",  # Table 23.5: two sub-heading lines
    }


def test_use_adult_dose_rows_link_to_a_real_adult_row(dosing: list[dict[str, str]], by_id: dict[str, dict]) -> None:
    adult_rows = [r for r in dosing if r["dose"] == "use adult dose"]
    assert len(adult_rows) == 12
    unresolved = set()
    for r in adult_rows:
        key = (r["chapter_no"], r["antibiotic"], r["weight_band"])
        assert r["frequency"] == "", key  # verbatim: the source prints no frequency here
        if not r["resolved_from"]:
            assert (r["resolved_dose"], r["resolved_frequency"]) == ("", ""), key
            unresolved.add(key)
            continue
        assert r["resolved_dose"] and r["resolved_frequency"], key
        chunk_id, antibiotic, route = r["resolved_from"].split(" | ")
        targets = [
            a for a in dosing
            if a["chunk_id"] == chunk_id and a["antibiotic"] == antibiotic and a["route"] == route
            and a["population"] == "adults" and not a["weight_band"]
        ]
        assert len(targets) == 1, (key, r["resolved_from"])  # the audit link is unambiguous
        adult = targets[0]
        assert adult["antibiotic"] == r["antibiotic"] and adult["chapter_no"] == r["chapter_no"], key
        assert adult["line"] == r["line"], key
        assert by_id[adult["chunk_id"]]["section_path"][-1] == by_id[r["chunk_id"]]["section_path"][-1], key  # same table
        assert (adult["dose"], adult["frequency"]) == (r["resolved_dose"], r["resolved_frequency"]), key
    assert unresolved == UNRESOLVED_ADULT_DOSE


def test_adult_dose_resolution_never_guesses() -> None:
    """Synthetic rows: two adult options are disambiguated by route, otherwise left empty and reported."""
    from src.parse.who_aware import DosingRow, Report, resolve_adult_doses

    def row(pop: str, dose: str, freq: str, route: str, band: str = "") -> DosingRow:
        values = dict(chapter_no="9", population=pop, antibiotic="Cloxacillin", line="other", dose=dose,
                      frequency=freq, route=route, weight_band=band, duration="5 days", chunk_id=f"c-{pop}")
        return DosingRow(values, values["chunk_id"], 1, table="Table 9.2", row_label="")

    iv, oral = row("adults", "2 g", "every 6 hours", "IV"), row("adults", "500 mg", "every 6 hours", "Oral")
    child = row("children", "use adult dose", "", "oral", "≥ 30 kg")
    report = Report()
    resolve_adult_doses([iv, oral, child], report)
    assert (child.values["resolved_dose"], child.values["resolved_from"]) == ("500 mg", "c-adults | Cloxacillin | Oral")

    second_oral = row("adults", "1 g", "every 8 hours", "Oral")
    child = row("children", "use adult dose", "", "oral", "≥ 30 kg")
    report = Report()
    resolve_adult_doses([iv, oral, second_oral, child], report)
    assert (child.values["resolved_dose"], child.values["resolved_frequency"], child.values["resolved_from"]) == ("", "", "")
    assert len(report.unresolved_adult_doses) == 1


def test_resolved_fields_on_other_rows(dosing: list[dict[str, str]]) -> None:
    for i, r in enumerate(dosing, start=2):
        if r["resolved_from"]:
            assert r["resolved_frequency"], i
        if r["dose"] != "use adult dose":
            assert (r["resolved_dose"], r["resolved_from"]) == (r["dose"], ""), i
            if r["frequency"]:
                assert r["resolved_frequency"] == r["frequency"], i


# --------------------------------------------------------------------------- #
# dose_basis and resolved_frequency rules
# --------------------------------------------------------------------------- #
def test_dose_basis_rules(dosing: list[dict[str, str]]) -> None:
    """dose_basis follows the printed wording: /day -> per_day; weekly -> course; single dose -> single."""
    for i, r in enumerate(dosing, start=2):
        dose = r["resolved_dose"] if r["dose"] == "use adult dose" else r["dose"]
        schedule = f"{r['frequency']} {r['duration']}".lower()
        if "/day" in dose:
            expected = "per_day"
        elif "dose per week" in schedule:
            expected = "course"
        elif "single dose" in schedule:
            expected = "single"
        else:
            expected = "per_dose"
        assert r["dose_basis"] == expected, (i, r["dose"], r["duration"])
    by = {(r["chapter_no"], r["antibiotic"], r["dose"], r["duration"][:17]): r["dose_basis"] for r in dosing}
    assert by[("23", "Amoxicillin+clavulanic acid", "80–90 mg/kg/day of amoxicillin component", "3–5 days")] == "per_day"
    assert by[("20", "Ceftriaxone", "250 mg", "Single dose")] == "single"
    assert by[("21", "Benzathine benzylpenicillin", "2.4 million IU (≈ 1.8 g)", "one dose per week")] == "course"
    assert by[("23", "Nitrofurantoin", "100 mg", "5 days")] == "per_dose"


def test_resolved_frequency_rules(dosing: list[dict[str, str]]) -> None:
    from src.parse.who_aware import RESOLVED_RULES

    not_stated = set()
    for i, r in enumerate(dosing, start=2):
        rule, freq = r["resolved_rule"], r["resolved_frequency"]
        assert rule in RESOLVED_RULES, i
        if rule == "as_printed":
            assert r["frequency"] and freq == r["frequency"], i
        elif rule == "adult_dose_lookup":
            assert r["dose"] == "use adult dose" and r["resolved_from"] and freq, i
        elif rule == "from_duration":
            assert not r["frequency"] and r["dose_basis"] != "per_day", i
            duration = r["duration"].lower()
            assert freq == ("once" if "single dose" in duration else "once weekly"), i
            assert "single dose" in duration or "one dose per week" in duration, i
        elif rule == "per_day_total":
            assert r["dose_basis"] == "per_day" and not r["frequency"] and freq == "", i
        elif rule == "not_stated":
            assert freq == "" and not r["frequency"], i
            not_stated.add((r["chapter_no"], r["antibiotic"], r["dose"]))
        else:  # "unresolved"
            assert r["dose"] == "use adult dose" and not freq and not r["resolved_from"], i
        # per_day rows never get a frequency.
        if r["dose_basis"] == "per_day" and r["dose"] != "use adult dose":
            assert freq == "", i
    assert not_stated == NOT_STATED_FREQUENCY


def test_pregnancy_population(dosing: list[dict[str, str]], chunks: list[dict]) -> None:
    for i, r in enumerate(dosing, start=2):
        assert ("pregnan" in r["population_raw"].lower()) == (r["population"] == "pregnant"), i
    pregnant_chunks = {c["chunk_id"]: c["population_raw"] for c in chunks if c["population"] == "pregnant"}
    assert pregnant_chunks == {
        "who-aware-2022-ch19-dosing-infection-in-pregnant-women-pregnant": "Infection in pregnant women",
        "who-aware-2022-ch21-dosing-syphilis-in-pregnancy-pregnant-other": "Syphilis in pregnancy",
        "who-aware-2022-ch21-dosing-syphilis-in-pregnancy-pregnant-allergy": "Syphilis in pregnancy",
    }


# --------------------------------------------------------------------------- #
# Verification samples
# --------------------------------------------------------------------------- #
def test_parser_run_never_touches_the_checked_sample(two_runs, tmp_path, monkeypatch) -> None:
    import src.parse.who_aware as wa

    first, _ = two_runs
    before = wa.VERIFIED_SAMPLE_MD.stat().st_mtime_ns if wa.VERIFIED_SAMPLE_MD.exists() else None
    for name in ("OUTPUT_PATH", "DOSING_CSV", "NO_ABX_CSV"):
        monkeypatch.setattr(wa, name, tmp_path / getattr(wa, name).name)
    monkeypatch.setattr(wa, "INTERIM_DIR", tmp_path)
    wa.write_outputs(first)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["aware.jsonl", "aware_dosing.csv", "aware_no_antibiotic.csv"]
    after = wa.VERIFIED_SAMPLE_MD.stat().st_mtime_ns if wa.VERIFIED_SAMPLE_MD.exists() else None
    assert before == after


def test_new_sample_never_overwrites_and_explains_empty_fields(two_runs, tmp_path) -> None:
    from src.parse.who_aware import DOSING_COLUMNS as COLS, empty_reason, write_sample

    first, _ = two_runs
    existing = tmp_path / "existing.md"
    existing.write_text("ticked by hand", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_sample(first, existing)
    assert existing.read_text(encoding="utf-8") == "ticked by hand"
    new = tmp_path / "new_sample.md"
    write_sample(first, new)
    text = new.read_text(encoding="utf-8")
    assert text.count("verified: [ ]") == 15 and "_(empty)_" not in text
    for row in first.dosing:
        for col in COLS:
            if not row[col]:
                assert empty_reason(row, col).startswith(("n/a: ", "MISSING: ")), (row["antibiotic"], col)
    adult = next(r for r in first.dosing if r["population"] == "adults" and not r["weight_band"])
    assert empty_reason(adult, "weight_band") == "n/a: no weight bands for adults"


def test_incomplete_rows_are_exactly_the_known_ones(dosing: list[dict[str, str]]) -> None:
    from src.parse.who_aware import missing_fields

    found = {}
    for r in dosing:
        missing = set(missing_fields(r))
        if missing:
            found[(r["chapter_no"], r["antibiotic"], r["dose"])] = missing
    assert found == KNOWN_INCOMPLETE_ROWS


def test_known_rows(dosing: list[dict[str, str]]) -> None:
    """Spot checks against Table 23.5 as printed (PDF pp. 306-307, printed pp. 290-291)."""
    rows = [r for r in dosing if r["chapter_no"] == "23"]
    nitro_adult = [r for r in rows if r["antibiotic"] == "Nitrofurantoin" and r["population"] == "adults"]
    assert {(r["dose"], r["frequency"], r["duration"]) for r in nitro_adult} == {
        ("100 mg", "every 12 hours", "5 days"), ("50 mg", "every 6 hours", "5 days"),
    }
    tmp_band = [r for r in rows if r["antibiotic"] == "Trimethoprim" and r["weight_band"] == "6–< 10 kg"]
    assert [(r["dose"], r["frequency"], r["aware_category"], r["printed_page"]) for r in tmp_band] == [
        ("40 mg", "every 12 hours", "Access", "291")
    ]
    # In-cell sub-groups narrow the population (Table 9.2 cloxacillin IV, printed p. 104).
    neonate = [r for r in dosing if r["chapter_no"] == "9" and r["population_raw"] == "Neonates"]
    assert [(r["antibiotic"], r["population"], r["dose"], r["frequency"]) for r in neonate] == [
        ("Cloxacillin", "neonates", "25–50 mg/kg/dose", "every 12 hours")
    ]


# --------------------------------------------------------------------------- #
# No-antibiotic content
# --------------------------------------------------------------------------- #
def test_no_antibiotic_csv(no_abx: list[dict[str, str]], by_id: dict[str, dict], chunks: list[dict]) -> None:
    assert no_abx, "no statements found"
    for r in no_abx:
        chunk = by_id[r["chunk_id"]]
        assert chunk["chunk_type"] == "no_antibiotic"
        assert quote_verifies(r["statement"], chunk) or r["statement"] in chunk["section_path"], r
    listed = {r["chunk_id"] for r in no_abx}
    for c in chunks:
        if c["chunk_type"] == "no_antibiotic":
            assert c["chunk_id"] in listed, c["chunk_id"]


def test_key_no_antibiotic_statements_present(no_abx: list[dict[str, str]]) -> None:
    statements = {(r["chapter_no"], r["statement"]) for r in no_abx}
    assert ("4", "Antibiotic treatment is not recommended for acute bronchitis and should be avoided.") in statements
    assert ("23", "A positive urine culture in asymptomatic patients is not an indication for antibiotic "
                  "treatment in the great majority of cases.") in statements


# --------------------------------------------------------------------------- #
# Source discrepancies and determinism
# --------------------------------------------------------------------------- #
def test_source_discrepancies_recorded(chunks: list[dict]) -> None:
    assert {c["chunk_id"] for c in chunks if c["source_discrepancy"]} == EXPECTED_DISCREPANCIES


@pytest.fixture(scope="module")
def two_runs():
    if not RAW_AVAILABLE:
        pytest.skip("raw EPUB/PDF not available")
    from src.parse.who_aware import parse

    return parse(), parse()


def test_deterministic_and_matches_files(two_runs, chunks: list[dict], dosing: list[dict[str, str]], no_abx) -> None:
    first, second = two_runs
    rec1 = [c.to_record() for c in first.chunks]
    rec2 = [c.to_record() for c in second.chunks]
    assert rec1 == rec2 == chunks
    assert first.dosing == second.dosing == dosing
    assert first.no_abx == second.no_abx == no_abx


def test_every_dose_also_found_in_pdf(two_runs) -> None:
    """EPUB vs PDF spot check: each dose and antibiotic string appears on its chunk's PDF pages."""
    first, _ = two_runs
    assert first.pdf_mismatches == []
