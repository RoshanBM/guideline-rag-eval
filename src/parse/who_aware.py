"""Parse the primary health care chapters of the WHO AWaRe antibiotic book (2022).

Outputs:

* ``data/chunks/aware.jsonl``: prose, ``no_antibiotic`` and table chunks (same
  schema as the hypertension chunks, plus ``infection``, ``population`` and
  ``context_header``).
* ``data/interim/aware_dosing.csv``: one row per antibiotic dosing option,
  values copied verbatim from the source.
* ``data/interim/aware_no_antibiotic.csv``: statements that antibiotics are not
  needed / not recommended.
* On request only (``--sample PATH``): a verification sample of 15 dosing rows
  (seed 42) for manual checking against the PDF, written to a *new* file.
  ``notes/aware_verification_sample.md`` (checked by hand) is never rewritten.

Run with::

    python -m src.parse.who_aware [--sample notes/aware_verification_sample_v2.md]

Source choice (see ``notes/source_inspection.md``): the EPUB is the primary
source because its dosing tables are real HTML tables and the AWaRe category is
a CSS class. The PDF is used only for page numbers (its own page labels) and to
cross-check every dose string. The infographic pages, which are JPEG-only in the
EPUB, are skipped and recorded.

Design notes:

* Footnote markers (``<sup>a</sup>``) are removed from body text so that quotes
  match cleanly; each chunk instead lists the footnotes its cells reference.
* Strength/certainty fields are always null: the AWaRe book has no graded
  recommendations.
* Nothing is guessed: unparseable or ambiguous dosing values are left empty and
  logged; inconsistencies in the source are recorded in ``source_discrepancy``.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pymupdf
from bs4 import BeautifulSoup, NavigableString, Tag

from src.parse.pdf_utils import page_label_map
from src.parse.schema import build_record, compose_text
from src.parse.sources import CHUNKS_DIR, LICENCE, PROJECT_ROOT, get_source
from src.parse.text_utils import normalise_text

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DOC_ID = "who-aware-2022"
DOC_TITLE = "The WHO AWaRe (Access, Watch, Reserve) antibiotic book (2022)"
COLLECTION = "aware"
# IRIS record for the book (found by ISBN search; see data/raw/MANIFEST.md).
SOURCE_URL = "https://iris.who.int/handle/10665/365237"
PART = "Primary health care"

OUTPUT_PATH = CHUNKS_DIR / "aware.jsonl"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DOSING_CSV = INTERIM_DIR / "aware_dosing.csv"
NO_ABX_CSV = INTERIM_DIR / "aware_no_antibiotic.csv"
# First verification sample, checked by hand. Never written by the parser again.
VERIFIED_SAMPLE_MD = PROJECT_ROOT / "notes" / "aware_verification_sample.md"
SAMPLE_SEED = 42
SAMPLE_SIZE = 15

MIN_WORDS = 150
MAX_WORDS = 400

AWARE_CLASSES = {"back_verde": "Access", "back_amarelo": "Watch", "back_vermelho": "Reserve"}

DOSING_COLUMNS = [
    "chapter_no", "infection", "population", "population_raw", "line", "antibiotic", "aware_category", "dose", "frequency",
    "dose_basis", "resolved_dose", "resolved_frequency", "resolved_rule", "resolved_from",
    "route", "duration", "weight_band", "dose_unit_note", "notes", "chunk_id", "pdf_page", "printed_page",
]
DOSE_BASES = ("per_dose", "per_day", "single", "course")
# How resolved_dose / resolved_frequency were obtained.
RESOLVED_RULES = {
    "as_printed": "frequency printed in the row, copied",
    "adult_dose_lookup": "row printed 'use adult dose'; dose and frequency copied from the adult row (resolved_from)",
    "from_duration": "no frequency printed; the duration states the schedule ('single dose' -> once, "
                     "'one dose per week' -> once weekly)",
    "per_day_total": "dose is a daily total (mg/kg/day); no frequency applies, so resolved_frequency is empty",
    "not_stated": "no frequency printed and none stated in the duration; resolved_frequency is empty",
    "unresolved": "row printed 'use adult dose' but no unique adult row was found; resolved fields are empty",
}
NO_ABX_COLUMNS = [
    "chapter_no", "infection", "section", "statement", "chunk_id",
    "pdf_page_start", "pdf_page_end", "printed_page_start", "printed_page_end",
]

# Subsections of "Laboratory tests". The EPUB markup sometimes tags them h2 (ch 23
# "Other tests") or as styled paragraphs (ch 10); they are normalised one level
# below "Laboratory tests".
LAB_SUBSECTIONS = {"Patient microbiology tests", "Other tests", "Using microbiology surveillance data"}

# Sentences stating that antibiotics are not needed / not recommended. Every match
# in the 20 chapters was reviewed by hand; see the parser summary for the list.
NO_ABX_PATTERNS = [
    r"\bantibiotics?(?: treatment| prophylaxis)? (?:is|are) not (?:\w+ )?(?:needed|recommended|required|necessary|appropriate|indicated)\b",
    r"\bno antibiotics? (?:is |are )?(?:needed|treatment)\b",
    r"\b(?:do|does|would) not (?:require|need|benefit from) (?:any )?antibiotic",
    r"\bno (?:need for|benefit from) (?:a course of )?antibiotic",
    r"\bnot (?:be )?(?:considered )?an indication for antibiotic",
    r"\bshould not be treated with antibiotics\b",
    r"\bantibiotics should not be used\b",
    r"\bwithout antibiotics is indicated\b",
    r"\broutine use of antibiotics to prevent infection[^.]*(?:should be avoided|is not required)",
    r"\bavoid the routine use of antibiotics\b",
    r"\bdoes not indicate [^.]*the need for antibiotics\b",
    r"\bsymptomatic treatment alone\b",
]
NO_ABX_RE = re.compile("|".join(f"(?:{p})" for p in NO_ABX_PATTERNS), re.IGNORECASE)
# Headings that make a whole section no-antibiotic content (ch 8).
NO_ABX_HEADING_RE = re.compile(r"\bdo not require antibiotic", re.IGNORECASE)

FOOTNOTE_MARK_RE = re.compile(r"^[a-z](?:,[a-z])*$")
SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z])")

# Known source discrepancies found while building and checking the parser. Keyed by
# chunk_id; the text is stored verbatim in that chunk's ``source_discrepancy``.
SOURCE_DISCREPANCIES: dict[str, str] = {
    "who-aware-2022-ch05-dosing-children-first-choice": (
        "Amoxicillin oral weight band printed as '3–6 kg' here (EPUB and PDF printed p. 44), whereas every other "
        "amoxicillin and amoxicillin+clavulanic acid weight-band table in the book, including the second-choice "
        "row of this table, prints '3–< 6 kg'. As printed, '3–6 kg' overlaps the next band ('6–< 10 kg') at 6 kg. "
        "Not resolved: the band is copied verbatim."
    ),
    "who-aware-2022-ch06-dosing-children-first-choice": (
        "Amoxicillin oral weight band printed as '3–6 kg' here (EPUB and PDF printed p. 58), whereas every other "
        "amoxicillin and amoxicillin+clavulanic acid weight-band table in the book prints '3–< 6 kg'. As printed, "
        "'3–6 kg' overlaps the next band ('6–< 10 kg') at 6 kg. Not resolved: the band is copied verbatim."
    ),
    "who-aware-2022-ch18-table-18-2": (
        "The EPUB omits this table's caption; the PDF (printed p. 223) prints 'Table 18.2 – Pathogens most "
        "frequently associated with bites'. Caption taken from the PDF."
    ),
}
# Caption missing from the EPUB, taken from the PDF (see SOURCE_DISCREPANCIES).
MISSING_CAPTIONS = {(18, 2): "Table 18.2 – Pathogens most frequently associated with bites"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def clean(el: Tag | None, keep_marks: bool = False) -> str:
    """Text of an element: footnote markers removed (or kept as plain letters), whitespace normalised."""
    if el is None:
        return ""
    el = copy.copy(el)
    for sup in el.find_all("sup"):
        t = sup.get_text(strip=True)
        if FOOTNOTE_MARK_RE.match(t) or not t:
            if keep_marks and t:
                sup.replace_with(t + " ")
            else:
                sup.decompose()
        else:
            sup.replace_with(t.translate(SUPERSCRIPT_DIGITS))
    text = re.sub(r"\s+", " ", el.get_text(""))
    # Remove the gap a deleted marker leaves before punctuation ("acid a (oral)", "IU a )").
    text = re.sub(r"\s+([),:;])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    return normalise_text(text).strip()


def footnote_marks(el: Tag) -> set[str]:
    marks: set[str] = set()
    for sup in el.find_all("sup"):
        t = sup.get_text(strip=True)
        if FOOTNOTE_MARK_RE.match(t):
            marks.update(t.split(","))
    return marks


def css_class(tag: Tag) -> str:
    value = tag.get("class", "")
    return " ".join(value) if isinstance(value, list) else str(value)


def slugify(text: str, max_words: int = 6) -> str:
    return "-".join(re.findall(r"[a-z0-9]+", text.lower())[:max_words])


def words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# EPUB loading and page tracking
# --------------------------------------------------------------------------- #
@dataclass
class Chapter:
    number: int
    title: str  # as printed, e.g. "23. Lower urinary tract infection"
    infection: str  # condition name, e.g. "Lower urinary tract infection"
    soup: BeautifulSoup
    href: str


def load_chapters(path: Path) -> list[Chapter]:
    """Load the spine documents strictly between the PRIMARY HEALTH CARE and HOSPITAL FACILITY dividers."""
    with zipfile.ZipFile(path) as zf:
        container = BeautifulSoup(zf.read("META-INF/container.xml"), "lxml-xml")
        opf_path = container.find("rootfile")["full-path"]
        opf_dir = PurePosixPath(opf_path).parent
        opf = BeautifulSoup(zf.read(opf_path), "lxml-xml")
        manifest = {item["id"]: item["href"] for item in opf.find_all("item")}
        hrefs = [str(opf_dir / manifest[r["idref"]]) for r in opf.find("spine").find_all("itemref")]
        names = [PurePosixPath(h).name for h in hrefs]
        start, end = names.index("Primary_Health_Care.xhtml"), names.index("HOSPITAL_FACILITY.xhtml")
        chapters = []
        for href in hrefs[start + 1 : end]:
            soup = BeautifulSoup(zf.read(href).decode("utf-8"), "lxml-xml")
            title = clean(soup.find("h1"))
            m = re.match(r"^(\d+)\.\s+(.*)$", title)
            if not m:
                raise ValueError(f"Unexpected chapter title in {href}: {title!r}")
            chapters.append(Chapter(int(m.group(1)), title, m.group(2), soup, href))
    return chapters


class PageIndex:
    """Printed page of every text node, from the ``epub:type="pagebreak"`` markers."""

    def __init__(self, soup: BeautifulSoup) -> None:
        self._page: dict[int, int] = {}
        current: int | None = None
        for node in soup.descendants:
            if isinstance(node, Tag):
                if node.name == "span" and str(node.get("id", "")).startswith("page_") and node.get("title"):
                    current = int(node["title"])
            elif isinstance(node, NavigableString) and node.strip() and current is not None:
                self._page[id(node)] = current

    def pages(self, *elements: Tag) -> tuple[int, int]:
        found = [self._page[id(s)] for el in elements if el is not None for s in el.find_all(string=True) if id(s) in self._page]
        if not found:
            raise ValueError("No page found for element")
        return min(found), max(found)


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    text: str
    page_start: int
    page_end: int
    no_abx: bool = False

    @property
    def words(self) -> int:
        return words(self.text)


@dataclass
class Leaf:
    chapter: Chapter
    path: list[str]
    blocks: list[Block] = field(default_factory=list)
    position: int = 0  # document order within the chapter
    no_abx_section: bool = False
    slug: str | None = None

    @property
    def words(self) -> int:
        return sum(b.words for b in self.blocks)


Population = tuple[str, str]  # (normalised value, verbatim label it was derived from)


@dataclass
class Chunk:
    chunk_id: str
    section_path: list[str]
    chunk_type: str  # "prose" | "table" | "no_antibiotic"
    source_text: str  # verbatim source only
    page_start: int  # printed page numbers (EPUB page markers = PDF page labels)
    page_end: int
    infection: str
    population: Population | None = None
    context_header: str | None = None  # parser-written, not source text
    source_discrepancy: str | None = None
    notes: list[str] = field(default_factory=list)  # verbatim table footnotes/notes the chunk uses
    aware_category_note: str | None = None  # derived from colour coding, not source text
    pdf_page_start: int | None = None
    pdf_page_end: int | None = None
    position: tuple[int, int, int] = (0, 0, 0)  # (chapter, document position, part) for ordering

    @property
    def text(self) -> str:
        """Retrieval text: context header + source text + Notes + AWaRe category note."""
        return compose_text(self.source_text, self.context_header, self.notes, self.aware_category_note)

    def to_record(self) -> dict[str, object]:
        """Record in the unified schema (src/parse/schema.py)."""
        return build_record(
            chunk_id=self.chunk_id,
            collection=COLLECTION,
            doc=DOC_TITLE,
            doc_id=DOC_ID,
            section_path=self.section_path,
            infection=self.infection,
            population=self.population[0] if self.population else None,
            population_raw=self.population[1] if self.population else None,
            context_header=self.context_header,
            chunk_type=self.chunk_type,
            index_default=True,
            source_discrepancy=self.source_discrepancy,
            source_text=self.source_text,
            aware_category_note=self.aware_category_note,
            text=self.text,
            pdf_page_start=self.pdf_page_start,
            pdf_page_end=self.pdf_page_end,
            printed_page_start=self.page_start,
            printed_page_end=self.page_end,
            source_url=SOURCE_URL,
            licence=LICENCE,
        )


@dataclass
class Report:
    skipped: Counter = field(default_factory=Counter)
    markup_fixes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    parse_log: list[str] = field(default_factory=list)
    no_abx_statements: list[str] = field(default_factory=list)
    unresolved_adult_doses: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Dosing tables
# --------------------------------------------------------------------------- #
@dataclass
class Para:
    """One paragraph or list item of a table cell."""

    text: str
    el: Tag
    drugs: list[tuple[str, str | None]]  # (name, AWaRe category) from coloured spans, in order
    marks: set[str]


@dataclass
class Regimen:
    dose: str
    frequency: str
    route: str
    weight_band: str
    notes: list[str]
    para: Para
    inline_duration: str = ""
    subgroup: str = ""  # in-cell sub-group label, e.g. "Neonates", "First week of life"
    unit_note: str = ""  # sub-heading over the band, e.g. "Oral weight bands; mg of sulfamethoxazole/trimethoprim component"


@dataclass
class Option:
    antibiotic: str
    category: str | None
    route: str
    line: str
    population: Population | None
    qualifier: str | None
    para: Para
    regimens: list[Regimen] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    combo: list[str] = field(default_factory=list)
    duration: "Duration" = field(default_factory=lambda: Duration())


ROUTES = {"oral", "iv", "im", "iv/im", "iv/oral", "eye drops", "eye ointment"}
# Population labels used as headings inside treatment cells, and in-cell sub-group labels.
# "Neonates and children" normalises to children (neonates are a subset of children).
POP_HEADERS = {"Adults": "adults", "Children": "children", "Neonates and children": "children",
               "Children and adults": "all", "Adults and children": "all"}
SUBGROUP_POPULATIONS = {"Neonates": "neonates", "First week of life": "neonates", "Children": "children",
                        "Adults": "adults"}
POPULATION_LABELS = {"adults": "adults", "adults_adolescents": "adults and adolescents", "children": "children",
                     "neonates": "neonates", "all": "adults and children", "pregnant": "pregnant women"}
LINE_MARKS = {"First choice": "first_choice", "Second choice": "second_choice"}
QUALIFIERS = {"Mild cases", "Severe cases", "Oral options", "Parenteral option", "Systemic treatment", "ADD",
              "Intravitreal injection", "Early syphilis", "Late syphilis or unknown stage"}
ALLERGY_RE = re.compile(r"^Alternative options? \(not in the EML\) in case of allergy to penicillin \(or stock-outs\):\s*(?P<rest>.*)$")
BAND_HEADER_RE = re.compile(r"^(?:Oral weight bands|mg of (?:the )?sulfamethoxazole/trimethoprim component)")
WEIGHT_BAND_RE = re.compile(
    r"^•?\s*(?P<band>(?:[≥<>]\s*)?\d+(?:\.\d+)?\s*(?:–\s*<?\s*\d+(?:\.\d+)?\s*)?kg(?:\s*\([^)]*\))?)\s*:?\s*(?P<rest>.+)$"
)
SUBGROUP_RE = re.compile(r"^•?\s*(?P<group>Neonates|Children|Adults|First week of life|Beyond first week of life)\s*:\s*(?P<rest>.+)$")
ROUTE_LINE_RE = re.compile(r"^•?\s*(?P<route>IV|IM|Oral|IV/IM)\s*:\s*(?P<rest>.*)$")
FREQ_RE = re.compile(
    r"(?:given\s+)?(?P<f>once a day \(on day (?:1|one)\) followed by .+?once a day(?: for \d+ days)?"
    r"|every hour for \d+ hours then every \d+ hours until healed"
    r"|every \d+(?:\s*(?:–|to)\s*\d+)? hours|every hour|once a day|twice a day|single dose)"
)
INLINE_DURATION_RE = re.compile(r"\s*\bfor (?P<d>\d+(?:\s*–\s*\d+)? (?:days?|weeks?))\b")
NOTE_PAREN_RE = re.compile(r"\s*\((?:maximum|or |modified|immediate)")


def balanced_paren_end(text: str, start: int) -> int:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def parse_regimen(text: str) -> tuple[str, str, str, list[str]]:
    """Split a regimen string into (dose, frequency, inline duration, notes), all verbatim substrings."""
    notes: list[str] = []
    text = text.strip()
    # Trailing explanatory parentheses, e.g. "(modified-release formulation)", "(maximum 1 g ...)".
    m = NOTE_PAREN_RE.search(text)
    while m:
        paren_start = text.index("(", m.start())
        end = balanced_paren_end(text, paren_start)
        notes.append(text[paren_start:end].strip())
        text = (text[: m.start()] + text[end:]).strip()
        m = NOTE_PAREN_RE.search(text)
    # A trailing ", but ..." clause is commentary.
    but = re.search(r",\s*but\b", text)
    if but:
        notes.append(text[but.start() + 1 :].strip())
        text = text[: but.start()].strip()
    freq = ""
    fm = FREQ_RE.search(text)
    if fm:
        dose_part, tail = text[: fm.start()], text[fm.end() :]
        freq = fm.group("f")
    else:
        dose_part, tail = text, ""
    inline = ""
    for part_name in ("tail", "dose"):
        part = tail if part_name == "tail" else dose_part
        dm = INLINE_DURATION_RE.search(part)
        if dm:
            inline = dm.group("d")
            part = (part[: dm.start()] + part[dm.end() :]).strip()
            if part_name == "tail":
                tail = part
            else:
                dose_part = part
            break
    dose_part = re.sub(r"[\s,]+$", "", re.sub(r"\s*given\s*$", "", dose_part.strip()))
    dose = dose_part if (re.search(r"\d", dose_part) or dose_part == "use adult dose") else ""
    if dose_part and not dose:
        notes.append(dose_part)
    tail = tail.strip(" ,")
    if tail:
        notes.append(tail)
    return dose, freq, inline, notes


def cell_paras(td: Tag) -> list[Para]:
    """Paragraphs and list items of a cell, in order (nested p inside li are part of the li)."""
    paras: list[Para] = []
    elements = [e for e in td.find_all(["p", "li"]) if not e.find_parent(["p", "li"]) or e.find_parent(["p", "li"]) not in td.find_all(["p", "li"])]
    if not elements:
        elements = [td]
    for el in elements:
        if el.name == "p" and el.find_parent("li") is not None and el.find_parent("li") in elements:
            continue
        text = clean(el)
        if not text:
            continue
        if el.name == "li":
            text = "• " + text
        drugs = [(clean(sp), AWARE_CLASSES.get(css_class(sp))) for sp in el.find_all("span", class_=re.compile(r"^back_"))]
        paras.append(Para(text, el, drugs, footnote_marks(el)))
    return paras


@dataclass
class TableGrid:
    caption: str
    caption_el: Tag | None
    number: tuple[int, int] | None
    callouts: list[tuple[str, list[str], Tag]]  # (title, texts, element)
    header: list[str]  # last header row (column roles for dosing tables)
    rows: list[list[Tag]]  # data rows
    foot: list[tuple[str | None, str, Tag]]  # (letter, text, element)
    ordered: list[tuple[bool, list[Tag | None]]] = field(default_factory=list)  # (is_header, row) in source order
    header_els: list[Tag | None] = field(default_factory=list)  # cells of the last header row


def table_grid(table: Tag, chapter_no: int) -> TableGrid:
    cap_el = table.find("caption")
    caption = clean(cap_el)
    m = re.match(r"^Table (\d+)\.(\d+)", caption)
    number = (int(m.group(1)), int(m.group(2))) if m else None
    raw_rows: list[Tag] = []
    for section in ("thead", "tbody"):
        sec = table.find(section)
        if sec is not None:
            raw_rows += sec.find_all("tr", recursive=False)
    if not raw_rows:
        raw_rows = table.find_all("tr")
    ncols = max(sum(int(td.get("colspan", 1)) for td in r.find_all(["td", "th"], recursive=False)) for r in raw_rows)
    # Expand rowspan/colspan into a grid of cell references.
    grid: list[list[Tag | None]] = []
    carry: dict[int, tuple[Tag, int]] = {}
    callouts: list[tuple[str, list[str], Tag]] = []
    header_rows: list[list[Tag]] = []
    ordered: list[tuple[bool, list[Tag | None]]] = []
    for r in raw_rows:
        cells = r.find_all(["td", "th"], recursive=False)
        if len(cells) == 1 and int(cells[0].get("colspan", 1)) == ncols and cells[0].find("p", class_="box_tit"):
            title = clean(cells[0].find("p", class_="box_tit"))
            texts = [clean(p) for p in cells[0].find_all("p") if "box_tit" not in css_class(p) and clean(p)]
            callouts.append((title, texts, cells[0]))
            continue
        row: list[Tag | None] = []
        col = 0
        it = iter(cells)
        while col < ncols:
            if col in carry:
                cell, left = carry[col]
                row.append(cell)
                if left > 1:
                    carry[col] = (cell, left - 1)
                else:
                    del carry[col]
                col += 1
                continue
            cell = next(it, None)
            if cell is None:
                row.append(None)
                col += 1
                continue
            span = int(cell.get("colspan", 1))
            rspan = int(cell.get("rowspan", 1))
            for k in range(span):
                row.append(cell)
                if rspan > 1:
                    carry[col + k] = (cell, rspan - 1)
            col += span
        is_header = r.parent.name == "thead" or all(c is not None and "colort" in css_class(c) for c in row)
        (header_rows if is_header else grid).append(row)
        ordered.append((is_header, row))
    header = [clean(c) if c is not None else "" for c in header_rows[-1]] if header_rows else [""] * ncols
    foot: list[tuple[str | None, str, Tag]] = []
    tfoot = table.find("tfoot")
    if tfoot is not None:
        for p in tfoot.find_all("p"):
            text = clean(p, keep_marks=True)
            if not text:
                continue
            fm = re.match(r"^([a-z])\s", text)
            foot.append((fm.group(1) if fm and p.find("sup") else None, text, p))
    if number is None:
        # Uncaptioned table: number it from its neighbours and take the caption from the PDF.
        prev = table.find_previous("caption")
        pm = re.match(r"^Table (\d+)\.(\d+)", clean(prev)) if prev else None
        guess = (chapter_no, int(pm.group(2)) + 1) if pm else None
        if guess in MISSING_CAPTIONS:
            number, caption = guess, MISSING_CAPTIONS[guess]
    header_els = list(header_rows[-1]) if header_rows else [None] * ncols
    return TableGrid(caption, cap_el, number, callouts, header, grid, foot, ordered, header_els)


def column_role(header: str) -> tuple[str, Population | None, str | None]:
    """Return (kind, population, line) for a column header; population is (normalised, header text)."""
    h = header.lower()
    if h.startswith("adults and children") or h.startswith("children and adults"):
        return "treat", ("all", header), None
    if h.startswith("adults"):
        return "treat", ("adults", header), None
    if h.startswith("children"):
        return "treat", ("children", header), None
    if "duration" in h:
        return "duration", None, None
    if h.startswith("first choice"):
        return "treat", None, "first_choice"
    if h.startswith("second choice"):
        return "treat", None, "second_choice"
    if h in ("treatment", "antibiotic treatment"):
        return "treat", None, None
    return "label", None, None


# (pattern, normalised population) in priority order; the matched text is kept verbatim
# as population_raw.
POPULATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # A label naming pregnancy ("Syphilis in pregnancy", "Infection in pregnant women") is kept
    # whole (up to any bracket) as population_raw.
    (re.compile(r"[^()]*\bpregnan\w*[^()]*", re.I), "pregnant"),
    (re.compile(r"children and adults|adults and children", re.I), "all"),
    (re.compile(r"adults,? (?:and )?adolescents", re.I), "adults_adolescents"),
    (re.compile(r"(?:ophthalmia )?neonat\w*|newborn", re.I), "neonates"),
    (re.compile(r"\binfants?\b", re.I), "children"),
]


def population_from_text(text: str) -> Population | None:
    """Population named in a row label or caption, as (normalised, verbatim matched label)."""
    for pattern, value in POPULATION_PATTERNS:
        m = pattern.search(text)
        if m:
            return value, m.group(0).strip()
    adult = re.search(r"\badults?\b", text, re.I)
    child = re.search(r"\bchild\w*", text, re.I)
    if adult and not child:
        return "adults", adult.group(0)
    if child and not adult:
        return "children", child.group(0)
    return None


@dataclass
class ParaState:
    """Which chunk group a cell paragraph belongs to, and the qualifier headings governing it."""

    para: Para
    population: Population | None
    line: str
    qualifiers: tuple[Para, ...]  # e.g. ("ADD", "Systemic treatment") or ("Early syphilis",)


@dataclass
class CellResult:
    options: list[Option]
    para_states: list[ParaState]
    notes: list[str]


def parse_cell(paras: list[Para], population: Population | None, line: str, antibiotics: set[str], log: list[str], where: str) -> CellResult:
    """Parse a treatment cell into options, tracking population/line markers inside the cell."""
    options: list[Option] = []
    states: list[ParaState] = []
    cell_notes: list[str] = []
    pop, ln = population, line
    qualifier: str | None = None
    qual_paras: tuple[Para, ...] = ()
    prev_was_qualifier = False
    route_ctx: str | None = None
    band_note: str | None = None  # sub-heading(s) over weight bands; goes to dose_unit_note, not notes
    band_note_index = -2  # paragraph index of the last band sub-heading (to join consecutive ones)
    cur: Option | None = None
    link_next = False

    def known_drug_at_start(text: str, para: Para) -> str | None:
        t = text.lstrip("• ").strip()
        for name, _ in para.drugs:
            if t.startswith(name):
                return name
        for name in sorted(antibiotics, key=len, reverse=True):
            if t.lower().startswith(name.lower()) and re.match(r"[\s(:]|$", t[len(name):len(name) + 1] or " "):
                return t[: len(name)]
        return None

    def add_regimen(opt: Option, text: str, para: Para, band: str = "", route: str = "", extra: list[str] | None = None,
                    subgroup: str = "", unit_note: str = "") -> None:
        pieces = [text]
        # "500 mg given every 8 hours or 1 g given every 12 hours" -> two regimens.
        split = re.split(r"\s+or\s+(?=\d)", text)
        if len(split) > 1 and all(FREQ_RE.search(s) for s in split):
            pieces = split
        for piece in pieces:
            dose, freq, inline, notes = parse_regimen(piece)
            opt.regimens.append(Regimen(dose, freq, route or route_ctx or opt.route, band, (extra or []) + notes, para,
                                        inline, subgroup, unit_note))

    def new_option(name: str, rest: str, para: Para) -> Option:
        nonlocal cur, link_next
        category = next((c for n, c in para.drugs if n == name), None)
        opt = Option(name, category, "", ln, pop, qualifier, para)
        rest = rest.strip()
        pm = re.match(r"^\((?P<p>[^()]*)\)\s*(?P<r>.*)$", rest)
        if pm and pm.group("p").lower() in ROUTES:
            opt.route = pm.group("p")
            rest = pm.group("r")
        elif pm:
            rest = pm.group("p") + " " + pm.group("r")  # "(2 g single dose)" is a regimen, not a route
        rest = rest.lstrip(": ").strip()
        rm = ROUTE_LINE_RE.match(rest)
        if rm:
            opt.route = rm.group("route")
            rest = rm.group("rest")
        trailing_and = bool(re.search(r"\s*\bAND$", rest))
        rest = re.sub(r"\s*\bAND$", "", rest).strip()
        if link_next and options:
            opt.combo.append(options[-1].antibiotic)
            options[-1].combo.append(opt.antibiotic)
        link_next = trailing_and
        if rest:
            add_regimen(opt, rest, para)
        options.append(opt)
        cur = opt
        return opt

    for p_index, para in enumerate(paras):
        is_qualifier = para.text in QUALIFIERS
        if is_qualifier:
            # Consecutive qualifier headings ("ADD", "Systemic treatment") apply together.
            qual_paras = (qual_paras + (para,)) if prev_was_qualifier else (para,)
        prev_was_qualifier = is_qualifier
        # Split "First week of life: ... Beyond first week of life: ..." into two parts.
        for t in re.split(r"\s+(?=Beyond first week of life:)", para.text):
            if t in POP_HEADERS:
                pop, cur, route_ctx, link_next = (POP_HEADERS[t], t), None, None, False
                continue
            if t in LINE_MARKS:
                ln, cur, route_ctx, link_next = LINE_MARKS[t], None, None, False
                continue
            am = ALLERGY_RE.match(t)
            if am:
                ln, cur, route_ctx, link_next = "allergy", None, None, False
                t = am.group("rest")
                if not t:
                    continue
            if t == "OR":
                cur, route_ctx, band_note, link_next = None, None, None, False
                continue
            if t == "AND":
                link_next = True
                continue
            if t in QUALIFIERS:
                qualifier, cur, ln, band_note, link_next = t, None, line, None, False
                route_ctx = "Intravitreal injection" if t == "Intravitreal injection" else None
                continue
            if BAND_HEADER_RE.match(t):
                # Consecutive sub-headings ("Oral weight bands:" then "mg of sulfamethoxazole/trimethoprim
                # component") are kept together, verbatim.
                heading = t.rstrip(":")
                band_note = f"{band_note}; {heading}" if band_note and band_note_index == p_index - 1 else heading
                band_note_index = p_index
                continue
            wm = WEIGHT_BAND_RE.match(t)
            if wm and cur is not None:
                # Bands listed under "Oral weight bands" are oral even when the drug line says "(IV/oral)".
                band_route = "Oral" if band_note and band_note.startswith("Oral weight bands") and "/" in cur.route else ""
                add_regimen(cur, wm.group("rest"), para, band=wm.group("band").strip(), route=band_route,
                            unit_note=band_note or "")
                continue
            sm = SUBGROUP_RE.match(t)
            if sm and cur is not None:
                add_regimen(cur, sm.group("rest"), para, extra=[sm.group("group")], subgroup=sm.group("group"))
                continue
            rm = ROUTE_LINE_RE.match(t)
            if rm and cur is not None:
                route_ctx = rm.group("route")
                if rm.group("rest"):
                    sub = SUBGROUP_RE.match(rm.group("rest"))
                    if sub:
                        add_regimen(cur, sub.group("rest"), para, route=route_ctx, extra=[sub.group("group")],
                                    subgroup=sub.group("group"))
                    else:
                        add_regimen(cur, rm.group("rest"), para, route=route_ctx)
                continue
            if t.startswith("• ") and cur is not None and not known_drug_at_start(t, para):
                add_regimen(cur, t[2:], para)
                continue
            # Drug lines; "X ... AND Probenecid (...)" splits into segments.
            segments = re.split(r"\s+AND\s+(?=[A-Z])", t)
            handled = False
            for k, seg in enumerate(segments):
                name = known_drug_at_start(seg, para)
                after = seg.lstrip("• ").strip()[len(name):] if name else ""
                bracket = re.match(r"^\s*\((?P<p>[^()]*)\)", after)
                if name and bracket and bracket.group("p").lower() not in ROUTES and re.search(r"\d", bracket.group("p")):
                    # "Azithromycin (2 g single dose) or erythromycin (500 mg ...)": several drugs, each with
                    # its regimen in brackets. Handled by the free-text path below so each drug keeps its own regimen.
                    break
                if name:
                    if k > 0:
                        link_next = True
                    new_option(name, seg.lstrip("• ").strip()[len(name):], para)
                    handled = True
                elif k > 0 and cur is not None:
                    cur.notes.append("AND " + seg)  # non-antibiotic adjunct, e.g. probenecid
                    handled = True
            if handled:
                continue
            # Free text naming drugs with a regimen in brackets: "Azithromycin (2 g single dose) or erythromycin (...)".
            hits = [m for m in re.finditer(r"\b(?P<n>[A-Za-z+ ]+?)\s*\((?P<r>[^()]*\d[^()]*)\)", t)
                    if m.group("n").strip().split(" ")[-1].lower() in {a.lower() for a in antibiotics}]
            if hits:
                remainder = t[hits[-1].end():].strip()
                for m in hits:
                    name = m.group("n").strip().split(" ")[-1]
                    opt = Option(name, None, "", ln, pop, qualifier, para)
                    add_regimen(opt, m.group("r"), para)
                    if remainder:
                        opt.notes.append(remainder)
                    options.append(opt)
                cur = None
                continue
            if t.endswith(".") or cur is None:
                # A full sentence applies to every option in the cell.
                cell_notes.append(t)
            else:
                cur.notes.append(t)
        # The state that applies to this paragraph's content (markers belong to the group they open).
        states.append(ParaState(para, pop, ln, qual_paras))
    for opt in options:
        if opt.combo:
            opt.notes.append("in combination with " + " and ".join(opt.combo))
        opt.notes += cell_notes
    # Marker paragraphs whose group has no options belong with the next group in the cell.
    with_options = {(o.population, o.line) for o in options}
    nxt: tuple[str | None, str] | None = None
    for st in reversed(states):
        if (st.population, st.line) not in with_options and nxt is not None:
            st.population, st.line = nxt
        nxt = (st.population, st.line)
    return CellResult(options, states, cell_notes)


@dataclass
class Duration:
    """Duration assigned to one option, with the footnote markers printed on the text it came from."""

    text: str = ""
    marks: set[str] = field(default_factory=set)
    from_column: bool = False  # True when taken from the table's duration column


@dataclass
class _DurationEntry:
    key: str
    population: str | None
    value: str
    paras: list[Para]


def duration_for(option: Option, paras: list[Para], n_options: int, antibiotics: set[str], log: list[str], where: str) -> Duration:
    """Duration for one option from the row's duration cell (verbatim), or empty if ambiguous.

    The returned markers are those printed on the duration-cell paragraphs actually used,
    so each footnote follows the duration it annotates.
    """
    if option.regimens and all(r.inline_duration for r in option.regimens):
        return Duration(option.regimens[0].inline_duration)
    if option.line == "allergy":
        # Allergy alternatives are written as free text with their own duration; the
        # duration column describes the main regimen of the row, so it is not reused.
        return Duration()
    if not paras:
        return Duration()

    def used(text: str, ps: list[Para]) -> Duration:
        return Duration(text, set().union(*(p.marks for p in ps)), from_column=True)

    entries: list[_DurationEntry] = []
    unkeyed: list[Para] = []
    key: str | None = None
    lower_names = {a.lower() for a in antibiotics}
    for para in paras:
        t = para.text
        km = re.match(r"^(?P<k>[^:]{2,60}?)\s*:\s*(?P<v>.*)$", t)
        if km and (km.group("k").lower() in lower_names or km.group("k") in QUALIFIERS):
            key = km.group("k")
            if km.group("v"):
                entries.append(_DurationEntry(key, None, km.group("v"), [para]))
            continue
        pm = re.match(r"^(?P<v>.+?)\s*\((?P<k>[^()]+)\)$", t)
        if pm and pm.group("k") in ("adults", "children") and key:
            entries.append(_DurationEntry(key, pm.group("k"), pm.group("v"), [para]))
            continue
        if pm and any(re.search(rf"\b{re.escape(n)}\b", pm.group("k"), re.I) for n in lower_names):
            entries.append(_DurationEntry(pm.group("k"), None, pm.group("v"), [para]))
            continue
        if t in QUALIFIERS:
            key = t
            continue
        if key is not None and entries and entries[-1].key == key:
            entries[-1].value = f"{entries[-1].value}; {t}"
            entries[-1].paras.append(para)
        elif key is not None:
            entries.append(_DurationEntry(key, None, t, [para]))
        else:
            unkeyed.append(para)
    if not entries:
        texts = [p.text for p in unkeyed]
        bare = len(unkeyed) == n_options > 1 and all(len(u) < 25 and "(" not in u for u in texts)
        if bare:
            log.append(f"{where}: {option.antibiotic}: duration cell lists {texts} for {n_options} options without saying which is which; left empty")
            return Duration()
        return used("; ".join(texts), unkeyed)

    def matches(k: str) -> bool:
        kl = k.lower()
        return kl == option.antibiotic.lower() or (option.qualifier is not None and k == option.qualifier) or (
            bool(re.search(rf"\b{re.escape(option.antibiotic.lower())}\b", kl))
            and not any(n != option.antibiotic.lower() and n in kl and option.antibiotic.lower() in n for n in lower_names)
        )

    cands = [e for e in entries if matches(e.key)]
    if len(cands) > 1:
        pop_value = option.population[0] if option.population else None
        cands = [e for e in cands if e.population in (None, pop_value)] or cands
    if len(cands) > 1:
        topical = option.route in ("eye drops", "eye ointment")
        cands = [e for e in cands if ("topical" in e.key.lower()) == topical] or cands
    if len(cands) == 1:
        return used(cands[0].value, cands[0].paras)
    if not cands and option.qualifier is None and all(e.key in QUALIFIERS for e in entries):
        return used("; ".join(p.text for p in paras), paras)  # option applies to every qualifier listed
    if not cands and unkeyed:
        return used("; ".join(p.text for p in unkeyed), unkeyed)
    log.append(f"{where}: {option.antibiotic}: could not match duration in {[p.text for p in paras]}; left empty")
    return Duration()


# --------------------------------------------------------------------------- #
# Chapter walk
# --------------------------------------------------------------------------- #
@dataclass
class DosingRow:
    values: dict[str, str]
    chunk_id: str
    printed_page: int
    table: str = ""  # table caption (not exported; used to resolve "use adult dose")
    row_label: str = ""  # table row label, e.g. "Low" in Table 15.2 (not exported)


class ChapterParser:
    """Walk one chapter, producing prose leaves, table chunks and dosing rows."""

    def __init__(self, chapter: Chapter, antibiotics: set[str], report: Report) -> None:
        self.ch = chapter
        self.pages = PageIndex(chapter.soup)
        self.antibiotics = antibiotics
        self.report = report
        self.headings: dict[int, str] = {}
        self.leaves: list[Leaf] = []
        self.tables: list[Chunk] = []
        self.dosing: list[DosingRow] = []
        self.position = 0
        self._leaf: Leaf | None = None
        self.condition_h2 = {
            clean(h2) for h2 in chapter.soup.find_all("h2")
            if (nxt := h2.find_next(["h2", "h3"])) is not None and nxt.name == "h3" and clean(nxt) == "Definition"
        }
        self.n_dosing_tables = sum(
            1 for t in chapter.soup.find_all("table") if t.find("span", class_=re.compile(r"^back_"))
        )

    # -- paths and leaves --------------------------------------------------- #
    @property
    def path(self) -> list[str]:
        return [PART, self.ch.title] + [self.headings[k] for k in sorted(self.headings)]

    def _close(self) -> None:
        if self._leaf is not None and self._leaf.blocks:
            self.leaves.append(self._leaf)
        self._leaf = None

    def _leaf_for_path(self) -> Leaf:
        if self._leaf is None:
            self.position += 1
            path = self.path
            flagged = any(NO_ABX_HEADING_RE.search(p) for p in path[2:])
            self._leaf = Leaf(self.ch, path, position=self.position, no_abx_section=flagged)
        return self._leaf

    def _add_block(self, el: Tag, text: str) -> None:
        if not text:
            return
        p0, p1 = self.pages.pages(el)
        flagged = bool(NO_ABX_RE.search(text))
        self._leaf_for_path().blocks.append(Block(text, p0, p1, flagged))

    def _heading(self, level: int, title: str) -> None:
        if title in LAB_SUBSECTIONS and "Laboratory tests" in self.headings.values():
            lab_level = next(k for k, v in self.headings.items() if v == "Laboratory tests")
            if level != lab_level + 1:
                self.report.markup_fixes.append(
                    f"ch{self.ch.number}: heading {title!r} tagged at level {level}, placed under 'Laboratory tests' (level {lab_level + 1})"
                )
            level = lab_level + 1
        self._close()
        self.headings = {k: v for k, v in self.headings.items() if k < level}
        self.headings[level] = title

    # -- walk ------------------------------------------------------------------ #
    def run(self) -> None:
        body = self.ch.soup.find("body")
        self._walk(body)
        self._close()

    def _walk(self, el: Tag) -> None:
        children = [c for c in el.children if isinstance(c, Tag)]
        i = 0
        while i < len(children):
            c = children[i]
            cls = css_class(c)
            if c.name == "h1":
                pass
            elif c.name in ("h2", "h3"):
                self._heading(int(c.name[1]), clean(c))
            elif c.name == "p" and cls.startswith("H4"):
                self._heading(4, clean(c))
            elif c.name == "p" and "Infographic-Title" in cls:
                self.report.skipped[f"ch{self.ch.number}: infographic title"] += 1
            elif c.name == "p" and re.search(r"•\s*Page \d of \d$", c.get_text(" ", strip=True)):
                self.report.skipped[f"ch{self.ch.number}: infographic page caption"] += 1
            elif c.name == "figure":
                kind = "content figure (image)" if c.find("figcaption") else "infographic / badge image"
                self.report.skipped[f"ch{self.ch.number}: {kind}"] += 1
            elif c.name == "div" and c.find("p", class_="box_tit", recursive=False):
                self._box(c)
            elif c.name == "table":
                self._table(c)
            elif c.name in ("ul", "ol"):
                for li in c.find_all("li", recursive=False):  # nested lists stay inside their item
                    self._add_block(li, "• " + clean(li))
            elif c.name == "p":
                self._add_block(c, clean(c))
            elif c.name in ("section", "div", "body"):
                self._walk(c)
            else:
                text = clean(c)
                if text:
                    self._add_block(c, text)
            i += 1

    def _box(self, div: Tag) -> None:
        title = clean(div.find("p", class_="box_tit", recursive=False))
        if title.startswith("Other relevant WHO resources"):
            self.report.skipped[f"ch{self.ch.number}: 'Other relevant WHO resources' reference list"] += 1
            return
        if title in ("Key messages", "Note"):
            saved = dict(self.headings)
            self._close()
            deepest = max(self.headings) if self.headings else 1
            self.headings[deepest + 1] = title
            for c in div.find_all(["p", "li"]):
                if "box_tit" in css_class(c) or c.find_parent("li") is not None and c.name == "p":
                    continue
                self._add_block(c, ("• " if c.name == "li" else "") + clean(c))
            self._close()
            self.headings = saved
            return
        # Other boxes (e.g. "Box 23.1 – ...") stay in the running text of their section.
        for c in div.find_all(["p", "li"]):
            if c.name == "p" and c.find_parent("li") is not None:
                continue
            self._add_block(c, ("• " if c.name == "li" else "") + clean(c))

    # -- tables --------------------------------------------------------------- #
    def _table(self, table: Tag) -> None:
        self.position += 1
        grid = table_grid(table, self.ch.number)
        if grid.number is None:
            raise ValueError(f"ch{self.ch.number}: table without caption or number")
        if table.find("span", class_=re.compile(r"^back_")):
            self._dosing_table(table, grid)
        else:
            self._plain_table(table, grid)

    def _foot_lines(self, grid: TableGrid, marks: set[str]) -> tuple[list[str], list[Tag]]:
        lines, els = [], []
        for letter, text, el in grid.foot:
            if text.startswith("Legend:"):
                continue  # colour legend; categories are written out explicitly instead
            if letter is None or letter in marks:
                lines.append(text)
                els.append(el)
        return lines, els

    def _plain_table(self, table: Tag, grid: TableGrid) -> None:
        a, b = grid.number  # type: ignore[misc]
        lines = [grid.caption]
        for title, texts, _ in grid.callouts:
            lines += [title, *texts]
        # Header rows are rendered where they occur: some tables (e.g. 6.2 Centor score) have
        # a second header row part-way down.
        for _, row in grid.ordered:
            # A cell spanning several columns appears once.
            cells = [
                "\n".join(p.text for p in cell_paras(cell)) if cell is not None else ""
                for k, cell in enumerate(row)
                if k == 0 or cell is None or cell is not row[k - 1]
            ]
            lines.append(" | ".join(cells))
        foot_lines, _ = self._foot_lines(grid, footnote_marks(table))
        is_symptomatic = bool(re.search(r"Medicines to (?:consider|control)", grid.caption))
        # Symptomatic-medicine tables give adult and child doses inside each cell; no heading names a
        # population, so population is left null. Other tables take it from the caption, if named.
        population = None if is_symptomatic else population_from_text(re.sub(r"^Table [\d.]+ – ", "", grid.caption))
        header = f"{self.ch.infection} - non-antibiotic medicines for symptomatic treatment:" if is_symptomatic else None
        p0, p1 = self.pages.pages(table)
        cid = f"{DOC_ID}-ch{self.ch.number:02d}-table-{a}-{b}"
        self.tables.append(
            Chunk(
                chunk_id=cid, section_path=self.path + [grid.caption], chunk_type="table",
                source_text=normalise_text("\n".join(lines)), page_start=p0, page_end=p1,
                infection=self.ch.infection, population=population, context_header=header,
                source_discrepancy=SOURCE_DISCREPANCIES.get(cid), notes=foot_lines,
                position=(self.ch.number, self.position, 0),
            )
        )

    def _dosing_table(self, table: Tag, grid: TableGrid) -> None:
        roles = [column_role(h) for h in grid.header]
        label_cols = [k for k, r in enumerate(roles) if r[0] == "label"]
        dur_cols = [k for k, r in enumerate(roles) if r[0] == "duration"]
        treat_cols = [k for k, r in enumerate(roles) if r[0] == "treat"]
        label_header = grid.header[label_cols[0]] if label_cols else ""
        dur_header = grid.header[dur_cols[0]] if dur_cols else ""
        dur_header_el = grid.header_els[dur_cols[0]] if dur_cols else None
        dur_header_marks = footnote_marks(dur_header_el) if dur_header_el is not None else set()
        caption_pop = population_from_text(re.sub(r"^Table [\d.]+ – ", "", grid.caption))
        equal_options = any("equal treatment options" in " ".join(t) for _, t, _ in grid.callouts)
        where = f"ch{self.ch.number} {grid.caption.split(' – ')[0]}"
        # Chapters covering several conditions (ch 10) have one h2 per condition.
        condition = self.headings.get(2) if self.headings.get(2) in self.condition_h2 else None

        groups: dict[tuple[str, Population | None, str], dict] = {}
        order: list[tuple[str, Population | None, str]] = []
        for r_index, row in enumerate(grid.rows):
            label_paras = cell_paras(row[label_cols[0]]) if label_cols and row[label_cols[0]] is not None else []
            label = label_paras[0].text if label_paras else ""
            row_line = LINE_MARKS.get(label, "other")
            row_label = "" if label in LINE_MARKS else label
            row_pop = population_from_text(" ".join(p.text for p in label_paras)) if label_paras else None
            dur_cell = row[dur_cols[0]] if dur_cols else None
            dur_paras = cell_paras(dur_cell) if dur_cell is not None else []
            for k in treat_cols:
                cell = row[k]
                if cell is None:
                    continue
                kind, col_pop, col_line = roles[k]
                paras = cell_paras(cell)
                if not paras or paras[0].text in ("–", "-"):
                    self.report.notes.append(f"{where}: '{grid.header[k]}' cell is empty ('–') for row {label or r_index + 1}")
                    continue
                pop0 = col_pop or row_pop or caption_pop
                line0 = col_line or row_line
                result = parse_cell(paras, pop0, line0, self.antibiotics, self.report.parse_log, where)
                for opt in result.options:
                    opt.duration = duration_for(opt, dur_paras, len(result.options), self.antibiotics, self.report.parse_log, where)
                    if opt.duration.from_column:
                        # A marker on the duration column header (e.g. Table 23.5 "duration^d")
                        # annotates every duration taken from that column.
                        opt.duration.marks |= dur_header_marks
                # Group paragraphs by (population, line) state for the chunks.
                for st in result.para_states:
                    key = (row_label, st.population, st.line)
                    if key not in groups:
                        groups[key] = {"rows": defaultdict(lambda: {"states": [], "options": []}), "col_header": grid.header[k]}
                        order.append(key)
                    entry = groups[key]["rows"][r_index]
                    entry["states"].append(st)
                    entry.update(label_paras=label_paras, dur_paras=dur_paras)
                for opt in result.options:
                    key = (row_label, opt.population, opt.line)
                    if key not in groups:
                        raise ValueError(f"{where}: option {opt.antibiotic} has no chunk group {key}")
                    groups[key]["rows"][r_index]["options"].append(opt)

        # Build one chunk per (row label, population, line).
        multi_line = len({k[2] for k in order}) > 1
        multi_label = len({k[0] for k in order}) > 1
        for key in order:
            row_label, pop, line = key
            g = groups[key]
            options = [o for e in g["rows"].values() for o in e["options"]]
            if not options:
                continue
            id_parts = [f"ch{self.ch.number:02d}", "dosing"]
            if self.n_dosing_tables > 1:
                id_parts.append(f"t{grid.number[0]}-{grid.number[1]}")  # type: ignore[index]
            if multi_label and row_label:
                id_parts.append(slugify(row_label, 5))
            id_parts.append(pop[0].replace("_", "-") if pop else "unspecified")
            if multi_line or line != "other":
                id_parts.append(line.replace("_", "-"))
            cid = f"{DOC_ID}-" + "-".join(id_parts)

            header_parts = [self.ch.infection]
            if condition:
                header_parts.append(condition)
            elif self.n_dosing_tables > 1:
                # Several dosing tables in one chapter (e.g. ch 14: diarrhoea vs cholera): name the
                # table's subject from its caption.
                subject = re.search(r"\bfor (.+)$", re.sub(r"^Table [\d.]+ – ", "", grid.caption))
                if subject:
                    header_parts.append(subject.group(1))
            if row_label:
                lab = row_label if words(row_label) > 3 or not label_header else f"{label_header}: {row_label}"
                header_parts.append(lab)
            header_parts.append(POPULATION_LABELS[pop[0]] if pop else "population not specified")
            header_parts.append({
                "first_choice": "first choice", "second_choice": "second choice", "allergy": "alternatives in case of penicillin allergy",
                "alternative": "alternative", "other": "antibiotic treatment" + (" (equal options)" if equal_options else ""),
            }[line])
            context_header = " - ".join(header_parts) + ":"

            lines = [grid.caption]
            elements: list[Tag] = [grid.caption_el] if grid.caption_el is not None else []
            for title, texts, el in grid.callouts:
                lines += [title, *texts]
                elements.append(el)
            # Allergy alternatives carry their own durations; the row's duration cell describes the main regimen.
            show_duration = bool(dur_header) and line != "allergy"
            cols = ([label_header] if label_cols else []) + [g["col_header"]] + ([dur_header] if show_duration else [])
            lines.append(" | ".join(cols))
            marks: set[str] = set()
            for r_index in sorted(g["rows"]):
                e = g["rows"][r_index]
                cells = []
                if label_cols:
                    cells.append("\n".join(p.text for p in e.get("label_paras", [])))
                    for p in e.get("label_paras", []):
                        marks |= p.marks
                # Treatment cell paragraphs, repeating the qualifier headings ("Early syphilis",
                # "ADD", "Systemic treatment") that govern them when those headings sit in another group.
                own = {id(st.para) for st in e["states"]}
                cell_lines: list[str] = []
                last_q: tuple[Para, ...] | None = None
                for st in e["states"]:
                    if st.qualifiers != last_q:
                        for q in st.qualifiers:
                            if id(q) not in own:
                                cell_lines.append(q.text)
                                elements.append(q.el)
                        last_q = st.qualifiers
                    cell_lines.append(st.para.text)
                    marks |= st.para.marks
                    elements.append(st.para.el)
                cells.append("\n".join(cell_lines))
                if show_duration:
                    cells.append("\n".join(p.text for p in e.get("dur_paras", [])))
                    for p in e.get("dur_paras", []):
                        marks |= p.marks
                        elements.append(p.el)
                lines.append(" | ".join(cells))
            # Footnotes referenced by the cells used, the column headers or the caption.
            for h_el in table.find_all(["th", "td"], class_="colort"):
                marks |= footnote_marks(h_el)
            if grid.caption_el is not None:
                marks |= footnote_marks(grid.caption_el)
            foot_lines, foot_els = self._foot_lines(grid, marks)
            elements += foot_els
            cats = []
            for o in options:
                if o.category and (o.antibiotic, o.category) not in cats:
                    cats.append((o.antibiotic, o.category))
            aware_note = (
                "AWaRe category (derived from the colour coding in the source): "
                + "; ".join(f"{n} – {c.upper()}" for n, c in cats)
            ) if cats else None
            p0, p1 = self.pages.pages(*elements)
            chunk = Chunk(
                chunk_id=cid, section_path=self.path + [grid.caption], chunk_type="table",
                source_text=normalise_text("\n".join(lines)), page_start=p0, page_end=p1,
                infection=self.ch.infection, population=pop, context_header=context_header,
                source_discrepancy=SOURCE_DISCREPANCIES.get(cid), notes=foot_lines, aware_category_note=aware_note,
                position=(self.ch.number, self.position, len(self.tables)),
            )
            self.tables.append(chunk)
            for o in options:
                self._dosing_rows(o, chunk, row_label, grid)

    def _dosing_rows(self, opt: Option, chunk: Chunk, row_label: str, grid: TableGrid) -> None:
        regimens = opt.regimens or [Regimen("", "", opt.route, "", [], opt.para)]
        for reg in regimens:
            notes = []
            if row_label:
                notes.append(f"Row: {row_label}")
            if opt.qualifier:
                notes.append(opt.qualifier)
            notes += [n for n in reg.notes if n]
            notes += opt.notes
            # Footnotes on the drug/regimen lines, and on the duration this row uses (its cell
            # paragraph and the duration column header) unless the regimen states its own duration.
            letters = opt.para.marks | reg.para.marks
            if not reg.inline_duration:
                letters |= opt.duration.marks
            for letter in sorted(letters):
                for fl, ftext, _ in grid.foot:
                    if fl == letter:
                        notes.append(f"Footnote {ftext}")
            if chunk.source_discrepancy and reg.weight_band.startswith("3–6"):
                notes.append("Source discrepancy: see chunk source_discrepancy")
            p0, _ = self.pages.pages(reg.para.el)
            # An in-cell sub-group ("• Neonates: ...", "First week of life: ...") narrows the population.
            population = opt.population
            if reg.subgroup in SUBGROUP_POPULATIONS:
                population = (SUBGROUP_POPULATIONS[reg.subgroup], reg.subgroup)
            values = {
                "chapter_no": str(self.ch.number),
                "infection": self.ch.infection,
                "population": population[0] if population else "",
                "population_raw": population[1] if population else "",
                "line": opt.line,
                "antibiotic": opt.antibiotic,
                "aware_category": opt.category or "",
                "dose": reg.dose,
                "frequency": reg.frequency,
                "route": reg.route or opt.route,
                "duration": reg.inline_duration or opt.duration.text,
                "weight_band": reg.weight_band,
                "dose_unit_note": reg.unit_note,
                "notes": "; ".join(dict.fromkeys(notes)),
            }
            self.dosing.append(DosingRow(values, chunk.chunk_id, p0, grid.caption, row_label))


# --------------------------------------------------------------------------- #
# Leaves -> prose / no_antibiotic chunks
# --------------------------------------------------------------------------- #
def split_no_abx(leaves: list[Leaf]) -> tuple[list[Leaf], list[Leaf]]:
    """Separate no-antibiotic content from prose, keeping each leaf's position."""
    prose, no_abx = [], []
    for leaf in leaves:
        if leaf.no_abx_section:
            no_abx.append(leaf)
            continue
        flagged = [b for b in leaf.blocks if b.no_abx]
        rest = [b for b in leaf.blocks if not b.no_abx]
        if flagged:
            no_abx.append(Leaf(leaf.chapter, leaf.path, flagged, leaf.position))
        if rest:
            prose.append(Leaf(leaf.chapter, leaf.path, rest, leaf.position))
    return prose, no_abx


def merge_small(leaves: list[Leaf], cross_section: bool) -> list[Leaf]:
    """Merge leaves under MIN_WORDS with an adjacent leaf of the same chapter.

    Preference: same top-level section (path[:3]); then, if ``cross_section``,
    any adjacent leaf in the chapter. The merged path is the common prefix and
    sub-headings below it are kept as text lines.
    """
    groups = [[l] for l in leaves]

    def size(g: list[Leaf]) -> int:
        return sum(l.words for l in g)

    changed = True
    while changed:
        changed = False
        for i, g in enumerate(groups):
            if size(g) >= MIN_WORDS:
                continue
            target = None
            for same_section in ((True, False) if cross_section else (True,)):
                for j in (i + 1, i - 1):
                    if not 0 <= j < len(groups):
                        continue
                    h = groups[j]
                    if h[0].chapter is not g[0].chapter:
                        continue
                    if same_section and h[0].path[:3] != g[0].path[:3]:
                        continue
                    if size(g) + size(h) <= MAX_WORDS:
                        target = j
                        break
                if target is not None:
                    break
            if target is None:
                continue
            lo, hi = min(i, target), max(i, target)
            groups[lo] = groups[lo] + groups[hi]
            del groups[hi]
            changed = True
            break
    merged = []
    for g in groups:
        if len(g) == 1:
            merged.append(g[0])
            continue
        common = g[0].path
        for l in g[1:]:
            k = 0
            while k < min(len(common), len(l.path)) and common[k] == l.path[k]:
                k += 1
            common = common[:k]
        blocks: list[Block] = []
        shown = list(common)
        for l in g:
            for depth in range(len(common), len(l.path)):
                if depth >= len(shown) or shown[depth] != l.path[depth]:
                    blocks.append(Block(l.path[depth], l.blocks[0].page_start, l.blocks[0].page_start))
            shown = list(l.path)
            blocks += l.blocks
        leaf = Leaf(g[0].chapter, common, blocks, g[0].position, g[0].no_abx_section)
        leaf.slug = "-".join(filter(None, [slugify(" ".join(g[0].path[2:]), 6) or "intro", "to", slugify(g[-1].path[-1], 4)]))
        merged.append(leaf)
    return merged


def split_blocks(blocks: list[Block]) -> list[list[Block]]:
    total = sum(b.words for b in blocks)
    if total <= MAX_WORDS:
        return [blocks]
    n = math.ceil(total / MAX_WORDS)
    target = total / n
    parts: list[list[Block]] = [[]]
    count = 0
    for b in blocks:
        if parts[-1] and (count + b.words > MAX_WORDS or (count >= target and len(parts) < n)):
            parts.append([])
            count = 0
        parts[-1].append(b)
        count += b.words
    return parts


def leaves_to_chunks(leaves: list[Leaf], chunk_type: str, used: Counter) -> list[Chunk]:
    chunks = []
    for leaf in leaves:
        ch = leaf.chapter
        slug = leaf.slug or slugify(" ".join(leaf.path[2:]), 8) or "intro"
        base = f"{DOC_ID}-ch{ch.number:02d}-" + ("noabx-" if chunk_type == "no_antibiotic" else "") + slug
        parts = split_blocks(leaf.blocks)
        for k, blocks in enumerate(parts, start=1):
            cid = base if len(parts) == 1 else f"{base}-p{k}"
            used[cid] += 1
            if used[cid] > 1:
                cid = f"{cid}-{used[cid]}"
            header = None
            if chunk_type == "no_antibiotic":
                # Full section path below the chapter, so e.g. ch 10 names the eye condition.
                header = " - ".join([ch.infection, *(leaf.path[2:] or ["overview"])]) + ":"
            chunks.append(
                Chunk(
                    chunk_id=cid, section_path=leaf.path, chunk_type=chunk_type,
                    source_text=normalise_text("\n".join(b.text for b in blocks)),
                    page_start=min(b.page_start for b in blocks), page_end=max(b.page_end for b in blocks),
                    infection=ch.infection, context_header=header, source_discrepancy=SOURCE_DISCREPANCIES.get(cid),
                    position=(ch.number, leaf.position, k),
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    chunks: list[Chunk]
    dosing: list[dict[str, str]]
    no_abx: list[dict[str, str]]
    report: Report
    pdf_mismatches: list[str]


def collect_antibiotic_names(chapters: list[Chapter]) -> set[str]:
    names = set()
    for ch in chapters:
        for sp in ch.soup.find_all("span", class_=re.compile(r"^back_")):
            names.add(clean(sp))
    return names


def parse(epub_path: Path | None = None, pdf_path: Path | None = None) -> Result:
    epub_path = epub_path or get_source("aware_epub").path
    pdf_path = pdf_path or get_source("aware_pdf").path
    chapters = load_chapters(epub_path)
    report = Report()
    antibiotics = collect_antibiotic_names(chapters)
    used: Counter = Counter()
    chunks: list[Chunk] = []
    dosing_rows: list[DosingRow] = []
    for ch in chapters:
        cp = ChapterParser(ch, antibiotics, report)
        cp.run()
        prose, no_abx = split_no_abx(cp.leaves)
        chunks += leaves_to_chunks(merge_small(prose, cross_section=True), "prose", used)
        chunks += leaves_to_chunks(merge_small(no_abx, cross_section=False), "no_antibiotic", used)
        for t in cp.tables:
            used[t.chunk_id] += 1
            if used[t.chunk_id] > 1:
                raise ValueError(f"Duplicate table chunk id {t.chunk_id}")
        chunks += cp.tables
        dosing_rows += cp.dosing
    chunks.sort(key=lambda c: c.position)

    # PDF page numbers from the PDF's own page labels.
    doc = pymupdf.open(pdf_path)
    labels = page_label_map(doc)
    for c in chunks:
        c.pdf_page_start = labels[str(c.page_start)]
        c.pdf_page_end = labels[str(c.page_end)]
    by_id = {c.chunk_id: c for c in chunks}
    unused = set(SOURCE_DISCREPANCIES) - set(by_id)
    if unused:
        raise ValueError(f"SOURCE_DISCREPANCIES refer to unknown chunks: {sorted(unused)}")

    for r in dosing_rows:
        r.values["chunk_id"] = r.chunk_id
    resolve_adult_doses(dosing_rows, report)
    dosing = []
    for r in dosing_rows:
        row = dict(r.values)
        row["printed_page"] = str(r.printed_page)
        row["pdf_page"] = str(labels[str(r.printed_page)])
        dosing.append({k: row[k] for k in DOSING_COLUMNS})

    no_abx_rows = []
    for c in chunks:
        if c.chunk_type != "no_antibiotic":
            continue
        body = c.source_text
        statements = [s for line in body.split("\n") for s in SENTENCE_SPLIT_RE.split(line) if NO_ABX_RE.search(s)]
        if not statements:  # whole-section flag (heading says "do not require antibiotic treatment")
            statements = [next(p for p in c.section_path if NO_ABX_HEADING_RE.search(p))]
        for s in statements:
            report.no_abx_statements.append(f"{c.chunk_id}: {s}")
            no_abx_rows.append({
                "chapter_no": str(int(c.chunk_id.split("-ch")[1][:2])), "infection": c.infection,
                "section": " > ".join(c.section_path[2:]), "statement": s.lstrip("• "), "chunk_id": c.chunk_id,
                "pdf_page_start": str(c.pdf_page_start), "pdf_page_end": str(c.pdf_page_end),
                "printed_page_start": str(c.page_start), "printed_page_end": str(c.page_end),
            })

    pdf_mismatches = cross_check_pdf(dosing, by_id, doc)
    return Result(chunks, dosing, no_abx_rows, report, pdf_mismatches)


USE_ADULT_DOSE = "use adult dose"


def adult_row_id(values: dict[str, str]) -> str:
    """Audit identifier of a dosing row: chunk_id | antibiotic | route."""
    return f"{values['chunk_id']} | {values['antibiotic']} | {values['route']}"


def dose_basis(values: dict[str, str]) -> str:
    """What the dose amount refers to, from the printed wording (in this order of precedence).

    * ``per_day``: the dose is a daily total ("80–90 mg/kg/day").
    * ``course``: a weekly schedule ("one dose per week for 3 consecutive weeks").
    * ``single``: "single dose" in the frequency or duration.
    * ``per_dose``: everything else (amount given at each administration).

    A "use adult dose" row is classified by the adult dose it resolves to.
    """
    dose = values["dose"] if values["dose"] != USE_ADULT_DOSE else values.get("resolved_dose", "")
    schedule = f"{values['frequency']} {values['duration']}".lower()
    if "/day" in dose:
        return "per_day"
    if "dose per week" in schedule:
        return "course"
    if "single dose" in schedule:
        return "single"
    return "per_dose"


def resolve_frequency(values: dict[str, str]) -> tuple[str, str]:
    """(resolved_frequency, resolved_rule) for a row that is not a 'use adult dose' row."""
    if values["frequency"]:
        return values["frequency"], "as_printed"
    if values["dose_basis"] == "per_day":
        return "", "per_day_total"
    duration = values["duration"].lower()
    if "single dose" in duration:
        return "once", "from_duration"
    if "one dose per week" in duration:
        return "once weekly", "from_duration"
    return "", "not_stated"


def resolve_adult_doses(rows: list[DosingRow], report: Report) -> None:
    """Fill dose_basis and the resolved_* fields on every row.

    Rows printed as "use adult dose" are linked to the adult row for the same
    antibiotic in the same chapter, table, table row and line of treatment, with no
    weight band. If several adult rows qualify (e.g. two routes or formulations),
    the route must match; if it is still not unique, the resolved fields stay empty
    and the row is reported. Other rows keep their printed dose; their frequency is
    copied, or taken from a schedule stated in the duration (see ``RESOLVED_RULES``).
    """
    for r in rows:
        v = r.values
        if v["dose"] != USE_ADULT_DOSE:
            v["dose_basis"] = dose_basis(v)
            frequency, rule = resolve_frequency(v)
            v.update(resolved_dose=v["dose"], resolved_frequency=frequency, resolved_rule=rule, resolved_from="")
            continue
        v.update(resolved_dose="", resolved_frequency="", resolved_rule="unresolved", resolved_from="")
        v["dose_basis"] = dose_basis(v)
        candidates = [
            a for a in rows
            if a.values["population"] == "adults" and not a.values["weight_band"]
            and a.values["chapter_no"] == v["chapter_no"] and a.table == r.table and a.row_label == r.row_label
            and a.values["antibiotic"] == v["antibiotic"] and a.values["line"] == v["line"]
            and a.values["dose"] and a.values["dose"] != USE_ADULT_DOSE
        ]
        if len(candidates) > 1:
            candidates = [a for a in candidates if a.values["route"].lower() == v["route"].lower()]
        where = f"ch{v['chapter_no']} {r.table.split(' – ')[0]} {v['antibiotic']} {v['weight_band']}"
        if len(candidates) != 1:
            report.unresolved_adult_doses.append(
                f"{where}: {len(candidates)} matching adult rows {[adult_row_id(a.values) for a in candidates]}; left empty"
            )
            continue
        adult = candidates[0].values
        if not adult["frequency"]:
            report.unresolved_adult_doses.append(f"{where}: adult row {adult_row_id(adult)} has no frequency; left empty")
            continue
        v.update(resolved_dose=adult["dose"], resolved_frequency=adult["frequency"], resolved_rule="adult_dose_lookup",
                 resolved_from=adult_row_id(adult))
        v["dose_basis"] = dose_basis(v)


def _squash(text: str) -> str:
    """Whitespace-free, lower-case form for comparing EPUB strings with PDF text."""
    return re.sub(r"\s+", "", text).lower()


def cross_check_pdf(dosing: list[dict[str, str]], by_id: dict[str, Chunk], doc: pymupdf.Document) -> list[str]:
    """Check that every dose string also appears in the PDF on the pages of its chunk."""
    cache: dict[tuple[int, int], str] = {}
    mismatches = []
    for row in dosing:
        c = by_id[row["chunk_id"]]
        key = (c.pdf_page_start, c.pdf_page_end)  # type: ignore[assignment]
        if key not in cache:
            cache[key] = _squash(" ".join(doc[p - 1].get_text() for p in range(key[0], key[1] + 1)))
        text = cache[key]
        for field_name in ("dose", "antibiotic"):
            value = row[field_name]
            if value and not re.search(_pdf_pattern(value), text):
                mismatches.append(f"{row['chunk_id']} {field_name} {value!r} not found on PDF pages {key[0]}–{key[1]}")
    return mismatches


def _pdf_pattern(value: str) -> str:
    """Regex for a squashed EPUB string in squashed PDF text.

    The PDF prints footnote letters inline ("800 000 IUa)"), so an optional
    footnote marker is allowed before closing brackets and commas.
    """
    marker = r"(?:[a-h](?:,[a-h])*)?"
    return "".join(marker + re.escape(ch) if ch in "),;" else re.escape(ch) for ch in _squash(value))


def write_outputs(result: Result) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        for c in result.chunks:
            fh.write(json.dumps(c.to_record(), ensure_ascii=False) + "\n")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for path, columns, rows in ((DOSING_CSV, DOSING_COLUMNS, result.dosing), (NO_ABX_CSV, NO_ABX_COLUMNS, result.no_abx)):
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    # Verification samples are written only on request (main --sample), never overwritten:
    # a reviewer's ticks must not be lost by a parser rerun.


def empty_reason(row: dict[str, str], field_name: str) -> str:
    """Why a CSV field is empty: 'n/a: ...' when empty by design, 'MISSING: ...' otherwise."""
    missing = set(missing_fields(row))
    if field_name in missing:
        return {
            "route": "MISSING: no route printed for this option",
            "aware_category": "MISSING: drug not colour-coded in the source",
            "frequency": "MISSING: no frequency printed",
            "duration": "MISSING: duration not printed or not assignable to this option",
            "dose": "MISSING: no dose printed",
        }.get(field_name, "MISSING")
    adult_dose = row["dose"] == USE_ADULT_DOSE
    basis = row.get("dose_basis", "")
    if field_name == "weight_band":
        if row["population"] in ("adults", "adults_adolescents", "pregnant"):
            return "n/a: no weight bands for adults"
        if "/kg" in row["dose"]:
            return "n/a: dose given per kg, not by weight band"
        return "n/a: no weight band printed for this regimen"
    if field_name == "dose_unit_note":
        return "n/a: no sub-heading above this dose"
    if field_name == "frequency":
        if adult_dose:
            return "n/a: printed as 'use adult dose' (see resolved_frequency)"
        if basis == "per_day":
            return "n/a: daily total (mg/kg/day); per-dose frequencies are in the weight-band rows"
        if basis in ("single", "course"):
            return "n/a: schedule stated in the duration"
        return "n/a: no frequency printed"
    if field_name == "resolved_frequency":
        return {"per_day_total": "n/a: daily total, no frequency applies",
                "not_stated": "n/a: no frequency printed or stated in the duration",
                "unresolved": "n/a: adult row could not be identified uniquely"}.get(row.get("resolved_rule", ""), "n/a")
    if field_name == "resolved_from":
        return "n/a: dose printed in this row (no adult-dose lookup)"
    if field_name in ("population", "population_raw"):
        return "n/a: no population named by the table column, row label or caption"
    if field_name == "notes":
        return "n/a: no row label, qualifier or footnote applies"
    if field_name == "duration":
        return "n/a: single dose" if basis == "single" else "n/a: no duration column in this table"
    return "n/a"


def write_sample(result: Result, path: Path) -> None:
    """Write a verification sample to ``path``; refuses to overwrite an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new file name (samples are never overwritten)")
    by_id = {c.chunk_id: c for c in result.chunks}
    rng = random.Random(SAMPLE_SEED)
    sample = sorted(rng.sample(range(len(result.dosing)), SAMPLE_SIZE))
    shown = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else path.name
    out = [
        "# AWaRe dosing: human verification sample",
        "",
        f"_Generated by `python -m src.parse.who_aware --sample {shown}`: "
        f"{SAMPLE_SIZE} rows of `data/interim/aware_dosing.csv` sampled with `random.Random({SAMPLE_SEED})`. "
        "Check each row against the PDF page given, then tick the box._",
        "",
        "PDF: `data/raw/9789240062382-eng.pdf` (PDF page = file page; printed page = number printed on the page).",
        "",
        "Empty fields are shown as `n/a: <reason>` when empty by design, or `MISSING: <reason>` when the source "
        "does not give the value. `resolved_rule` values: "
        + "; ".join(f"`{k}` = {v}" for k, v in RESOLVED_RULES.items()) + ".",
        "",
    ]
    for n, idx in enumerate(sample, start=1):
        row = result.dosing[idx]
        c = by_id[row["chunk_id"]]
        out += [
            f"## {n}. CSV row {idx + 2}: {row['antibiotic']} ({row['infection']}, {row['population'] or 'population not specified'})",
            "",
            f"**PDF page {row['pdf_page']} / printed page {row['printed_page']}** (chunk spans PDF pages "
            f"{c.pdf_page_start}–{c.pdf_page_end})",
            "",
            "| field | value |",
            "|---|---|",
            *[f"| {k} | {(row[k] or '_' + empty_reason(row, k) + '_').replace('|', '/')} |" for k in DOSING_COLUMNS],
            "",
            f"Chunk `{c.chunk_id}`:",
            "",
            "```text",
            c.text,
            "```",
            "",
            "verified: [ ]",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as fh:  # "x": fail rather than overwrite
        fh.write("\n".join(out))


def missing_fields(row: dict[str, str]) -> list[str]:
    """Fields that are empty where a value would be expected (see summary for rules)."""
    missing = [f for f in ("antibiotic", "dose", "route", "aware_category") if not row[f]]
    single = "single dose" in (row["frequency"] + row["duration"]).lower()
    # A frequency is not expected for a daily total ("80–90 mg/kg/day"), "use adult dose", a single
    # intravitreal injection, or when the duration itself states the schedule ("one dose per week ...").
    frequency_expected = not (
        single or "/day" in row["dose"] or row["dose"] == "use adult dose"
        or row["route"] == "Intravitreal injection" or "per week" in row["duration"].lower()
    )
    if not row["frequency"] and frequency_expected:
        missing.append("frequency")
    if not row["duration"] and not single:
        missing.append("duration")
    return missing


def print_summary(result: Result) -> None:
    chunks = result.chunks
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("Chunks by type: " + ", ".join(f"{t}={n}" for t, n in sorted(Counter(c.chunk_type for c in chunks).items())))
    w = [words(c.text) for c in chunks]
    print(f"Words per chunk: min={min(w)} median={statistics.median(w):g} max={max(w)}")
    for t in sorted({c.chunk_type for c in chunks}):
        wt = [words(c.text) for c in chunks if c.chunk_type == t]
        print(f"  {t:<14} n={len(wt):<4} min={min(wt)} median={statistics.median(wt):g} max={max(wt)}")
    print("Chunks by infection (prose / no_antibiotic / table):")
    by_inf: dict[str, Counter] = defaultdict(Counter)
    for c in chunks:
        by_inf[c.infection][c.chunk_type] += 1
    for inf, cnt in by_inf.items():
        print(f"  {inf:<75} {cnt['prose']:>3} / {cnt['no_antibiotic']:>2} / {cnt['table']:>2}")
    print(f"Dosing CSV rows: {len(result.dosing)} -> {DOSING_CSV.relative_to(PROJECT_ROOT)}")
    per_ch = Counter(r["chapter_no"] for r in result.dosing)
    print("  rows per chapter: " + ", ".join(f"ch{k}={v}" for k, v in sorted(per_ch.items(), key=lambda x: int(x[0]))))
    miss = [(i, missing_fields(r)) for i, r in enumerate(result.dosing)]
    miss = [(i, m) for i, m in miss if m]
    print(f"  rows with missing fields: {len(miss)}  by field: {dict(Counter(f for _, m in miss for f in m))}")
    for i, m in miss:
        r = result.dosing[i]
        print(f"    row {i + 2}: ch{r['chapter_no']} {r['antibiotic']!r} dose={r['dose']!r} -> missing {m}")
    adult = [r for r in result.dosing if r["dose"] == USE_ADULT_DOSE]
    print(f"'{USE_ADULT_DOSE}' rows: {len(adult)}, resolved from the adult row: {sum(1 for r in adult if r['resolved_from'])}")
    for line in result.report.unresolved_adult_doses or ["unresolved: none"]:
        print(f"  - {line}")
    print(f"No-antibiotic statements: {len(result.no_abx)} -> {NO_ABX_CSV.relative_to(PROJECT_ROOT)}")
    print("Parse log (ambiguities left empty / unattached text):")
    for line in result.report.parse_log or ["none"]:
        print(f"  - {line}")
    print("PDF cross-check of dose and antibiotic strings:")
    for line in result.pdf_mismatches or ["all dosing rows found on their PDF pages"]:
        print(f"  - {line}")
    print("Source discrepancies recorded:")
    for c in chunks:
        if c.source_discrepancy:
            print(f"  - {c.chunk_id}: {c.source_discrepancy}")
    print("Markup fixes:")
    for line in result.report.markup_fixes:
        print(f"  - {line}")
    print("Skipped content:")
    for k, v in sorted(result.report.skipped.items()):
        print(f"  - {k}: {v}")
    print("Notes:")
    for line in result.report.notes:
        print(f"  - {line}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=Path, help="also write a new verification sample to this (new) file")
    args = ap.parse_args(argv)
    result = parse()
    write_outputs(result)
    print_summary(result)
    if args.sample:
        path = args.sample if args.sample.is_absolute() else PROJECT_ROOT / args.sample
        write_sample(result, path)
        print(f"Wrote verification sample to {path}")


if __name__ == "__main__":
    main()
