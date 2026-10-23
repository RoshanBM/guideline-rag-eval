"""Parse WHO HEARTS-D (diagnosis and management of type 2 diabetes) into chunks.

Input: ``data/raw/WHO-UCN-NCD-20.1-eng.pdf`` (see ``sources.py``), WHO's original 35-page PDF
(Adobe InDesign CS6, created 21 April 2020). Output: ``data/chunks/hearts_d.jsonl`` in the unified
schema (``schema.py``).

Layout facts this parser relies on (all measured; see ``notes/source_inspection.md``):

* Body text is HelveticaNeue 11 pt, colour ``BLACK``; flowcharts are MyriadPro. The text layer
  is clean Unicode (no Private Use Area glyphs), with three quirks handled in
  :func:`clean_spans`: every bullet is present twice (a StoneSans glyph and a "•" in the
  text), sub-bullets are a StoneSans "o" plus a Wingdings3 decoration, and table cells end
  with a line-break character inside the span.
* Headings are recognised by size and colour (:func:`heading_level`).
* Side boxes ("Risk factors for ...", "Metformin", ...) are ``BOX_GREY`` regular/medium text
  laid out beside or between body paragraphs; captions are ``BOX_GREY`` bold.
* Footnotes are ``GREY`` 9 pt at the foot of the page; their markers are superscript digits.
* Tables have horizontal ruling lines per column; rows and columns are read from them.
* Fig. 1, Fig. 2 and Annex 1 are flowcharts drawn as rounded boxes; their text is read box
  by box in a fixed column order (``FLOWCHARTS``).
* The printed page number is in the footer of every content page.

Skipped (recorded in the run summary): cover, copyright, contents, acknowledgements and
abbreviations (PDF pp. 1-7), the reference lists ("6 Resources", p. 28, and the final
"Resources" of Annex 2, p. 35), and the photographs of Figs 3-11 (only their captions are
text, so there is nothing to chunk).

Values are never corrected in ``source_text``: internal inconsistencies of the source are
attached to the chunks concerned as ``source_discrepancy`` (``DISCREPANCIES``), and the
documented typo corrections (``text_utils.TEXT_CORRECTIONS``) are applied to the retrieval
``text`` only and listed in ``text_corrections``.

Flowchart chunks also carry ``flow_text``, and the dose statements are exported to
``data/interim/hearts_d_doses.csv``; both are hand-transcribed from the figures and kept in
``hearts_d_derived.py``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

import dataclasses

from src.parse.pdf_utils import Span, extract_spans
from src.parse.hearts_d_derived import DOSE_STATEMENTS, FLOW_TEXT
from src.parse.schema import build_record, compose_corrected_text
from src.parse.sources import CHUNKS_DIR, LICENCE, PROJECT_ROOT, get_source
from src.parse.text_utils import apply_corrections, build_vocabulary, join_lines, normalise_for_match, normalise_text

DOC_ID = "who-hearts-d-2020"
DOC_TITLE = "HEARTS D: diagnosis and management of type 2 diabetes (WHO/UCN/NCD/20.1)"
COLLECTION = "cardiometabolic"
SOURCE_URL = "https://www.who.int/publications/i/item/WHO-UCN-NCD-20.1"
OUTPUT_PATH = CHUNKS_DIR / "hearts_d.jsonl"
DOSES_PATH = PROJECT_ROOT / "data" / "interim" / "hearts_d_doses.csv"
DOSE_COLUMNS = ["drug", "drug_as_printed", "drug_source", "dose", "dose_corrected", "dose_basis", "route",
                "frequency", "timing", "dosing_as_printed", "figure_or_section", "step", "branch", "condition",
                "chunk_id", "pdf_page", "printed_page", "source_discrepancy"]
HEADER_PREFIX = "HEARTS-D"

MIN_WORDS = 150
MAX_WORDS = 400

BLUE = 0x324099
BLACK = 0x231F20
WHITE = 0xFFFFFF
GREY = 0x808285  # footer, footnotes, level-4 headings
BOX_GREY = 0x58595B  # side boxes (regular/medium) and captions (bold)
DECORATION_FONTS = ("Wingdings3",)  # drawn decorations next to sub-bullets; no text content

FIRST_CONTENT_PAGE = 8  # pp. 1-7: cover, copyright, contents, acknowledgements, abbreviations
FOOTER_TOP = 790.0
LINE_TOLERANCE = 3.2  # max distance between span centres on one visual line
PARAGRAPH_GAP = 16.0  # line pitch is 14.5 inside a paragraph, >= 17.3 between paragraphs/items
SKIPPED_HEADINGS = {"6 Resources": "reference list", "Resources": "reference list"}
ITEM_START_RE = re.compile(r"^(•|o |[a-c]\) )")
SUPERSCRIPTS = {"2": "²", "3": "³"}


@dataclass(frozen=True)
class FlowchartSpec:
    """Reading order for one flowchart page: boxes are read column by column.

    ``columns`` are x positions; a box belongs to the first column whose boundary is greater
    than the box's left edge. Within a column, boxes are read top to bottom, then left to right.
    """

    caption_prefix: str | None  # caption that starts the figure; None = figure under a heading
    columns: tuple[float, ...]
    order_note: str


FLOWCHARTS: dict[int, FlowchartSpec] = {
    16: FlowchartSpec("Fig. 1", (300.0,), "left column, then right column"),
    25: FlowchartSpec("Fig. 2", (180.0, 385.0), "left column, then middle column, then right column"),
    29: FlowchartSpec(None, (), "top to bottom"),
}


@dataclass(frozen=True)
class Discrepancy:
    """An internal inconsistency of the source, attached to every chunk containing an anchor."""

    key: str
    description: str
    anchors: tuple[str, ...]


DISCREPANCIES: list[Discrepancy] = [
    Discrepancy(
        "gliclazide-unit",
        "Fig. 1 prints 'GLICLAZIDE 80 g 2 x daily'; every other gliclazide dose in the document is "
        "in mg (Fig. 1 itself: '160 mg gliclazide' as the twice-daily total, and 'gliclazide 80 mg "
        "2 x daily'; Fig. 2: 'gliclazide 80 mg bid'). Kept as printed; not corrected.",
        ("GLICLAZIDE 80 g 2 x daily",),
    ),
    Discrepancy(
        "screening-bmi",
        "Screening criterion for asymptomatic adults: Fig. 1 '40+ years old and BMI ≥25' versus "
        "Fig. 2 '40+ years old with BMI >25' (a BMI of exactly 25 qualifies only under Fig. 1).",
        ("40+ years old and BMI ≥25", "40+ years old with BMI >25"),
    ),
    Discrepancy(
        "hyperglycaemia-18",
        "Threshold for marked hyperglycaemia: '≥18 mmol/L' in the text (Diagnostic tests, p. 13; "
        "urgent referral criteria, p. 26) versus '>18 mmol/L' in Fig. 1 and Fig. 2 (Fig. 1's "
        "routine branch is 'FPG ≥7 mmol/L and <18 mmol/L').",
        (
            "If plasma glucose ≥18 mmol/L",
            "no improvement in glycaemia ≥18 mmol/L",
            "FPG/RPG >18 mmol/L",
            "FPG/RPG >18 mmol/l",
            "glucose >18 mmol/l (325 mg/dl)",
        ),
    ),
    Discrepancy(
        "hyperglycaemia-treatment",
        "Treatment for FPG/RPG >18 mmol/L (or symptomatic FPG ≥15 mmol/L) with urine ketones <2+: "
        "Fig. 1 gives metformin 1000 mg 2 x daily plus gliclazide ('80 g 2 x daily' as printed), "
        "whereas Fig. 2 begins gliclazide 80 mg bid only, without metformin.",
        ("GIVE METFORMIN 1000 mg 2 x daily", "BEGIN gliclazide 80 mg bid"),
    ),
    Discrepancy(
        "glycaemic-targets",
        "Glycaemic targets use different comparison signs: HbA1c '7.0%' (p. 17), '≤7.0%' (p. 27) and "
        "'<7.0%' (Table 7); FPG '≤7.0 mmol/L' (pp. 17 and 27; Fig. 2 '≤7.0 mmol/l') versus "
        "'<7.0 mmol/L' (Table 7); postprandial PG '≤9.0' (pp. 17 and 27) versus '<9.0' (Table 7). "
        "A value of exactly 7.0 is on target under one wording and not under the other.",
        (
            "aim for an HbA1c of 7.0%",
            "aim for a HbA1c of ≤7.0%",
            "HbA1c <7.0% (53 mmol/mol)",
            "Fasting | ≤7.0 mmol/l",
        ),
    ),
    Discrepancy(
        "urine-ketones",
        "Urine ketone threshold for referral: '≥2+' (Fig. 1 'Urine ketones ≥2+'; Fig. 2 'if ketones "
        "≥2+'), 'urine ketone 2+' (Fig. 2, severe hyperglycaemia) and '>2+' (urgent referral "
        "criteria, p. 26); a result of exactly 2+ is a referral under the first two only.",
        ("Urine ketones ≥2+", "if ketones ≥2+", "urine ketone 2+", "urine ketones >2+"),
    ),
    Discrepancy(
        "severe-hypoglycaemia",
        "Severe hypoglycaemia threshold: 'plasma glucose ≤2.8 mmol/L (50 mg/dL)' (p. 19) versus "
        "'plasma glucose <50 mg/dl or 2.8 mmol/l' (Fig. 2).",
        ("plasma glucose ≤2.8 mmol/L (50 mg/dL)", "<50 mg/dl or 2.8 mmol/l"),
    ),
    Discrepancy(
        "figure-references",
        "Figure cross-references in Annex 2 do not match the captions: the monofilament text cites "
        "'(Fig. 6 and Fig. 7)', but Fig. 6 is 'Posterior tibial artery palpation' (Fig. 7 is 'Sites "
        "to be tested for loss of protective sensation', Fig. 8 the monofilament test); the tuning "
        "fork text cites '(Fig. 8)', but the tuning fork is Fig. 9. Kept as printed.",
        ("three sites on both feet (Fig. 6 and Fig. 7)", "constant pressure (Fig. 8)"),
    ),
    Discrepancy(
        "first-review-interval",
        "First review before starting metformin: Fig. 1 reviews after 1 month ('REVIEW IN 1 MONTH (or "
        "immediately if symptoms appear)', then 'IF goal not achieved BEGIN METFORMIN 500 mg daily'), whereas "
        "Fig. 2 reviews after 3 months ('REVIEW IN 3 MONTHS If goal not achieved BEGIN METFORMIN 500 mg once "
        "daily').",
        ("REVIEW IN 1 MONTH (or immediately if symptoms appear)", "REVIEW IN 3 MONTHS If goal not achieved BEGIN"),
    ),
    Discrepancy(
        "table3-footnote",
        "Table 3: the note '* Overnight fast of 8–14 hours.' has no '*' marker in the table body "
        "(the fasting row carries only '**'); it presumably refers to the fasting row, but this is "
        "not stated.",
        ("Table 3 Diagnostic criteria for diabetes",),
    ),
]


# --------------------------------------------------------------------------- #
# Lines
# --------------------------------------------------------------------------- #


def is_bold(span: Span) -> bool:
    return "Bold" in span.font or span.font.endswith("-B")


def is_italic(span: Span) -> bool:
    return "Italic" in span.font or span.font.endswith("-I")


def is_bullet_glyph(span: Span) -> bool:
    return span.text.strip() == "•"


def clean_spans(spans: list[Span]) -> list[Span]:
    """Drop decorations and empty spans; line-break characters inside a span become spaces."""
    out = []
    for span in spans:
        if span.font in DECORATION_FONTS or not span.horizontal:
            continue
        text = re.sub(r"[\r\n\u2028\u2029]", " ", span.text)
        if text.strip():
            out.append(dataclasses.replace(span, text=text))
    return out


def _centre(span: Span) -> float:
    return (span.y0 + span.y1) / 2


@dataclass
class TLine:
    """A visual line built from spans whose vertical centres are within ``LINE_TOLERANCE``."""

    spans: list[Span]
    text: str
    markers: list[int]  # footnote numbers whose superscript markers were removed from ``text``

    @property
    def page(self) -> int:
        return self.spans[0].page

    @property
    def main(self) -> Span:
        return max(self.spans, key=lambda s: (s.size, len(s.text.strip())))

    @property
    def x0(self) -> float:
        return min(s.x0 for s in self.spans)

    @property
    def x1(self) -> float:
        return max(s.x1 for s in self.spans)

    @property
    def y0(self) -> float:
        return min(s.y0 for s in self.spans)

    @property
    def y1(self) -> float:
        return max(s.y1 for s in self.spans)

    @property
    def starts_item(self) -> bool:
        return bool(ITEM_START_RE.match(self.text))


def line_text(spans: list[Span], keep_superscripts: bool = False) -> tuple[str, list[int]]:
    """Text of one visual line (spans ordered left to right).

    A space is inserted where there is a visible gap. A bullet glyph becomes ``"• "`` (the
    original draws each bullet twice, so a repeat is dropped). A superscript digit directly
    after a unit ``m`` is an exponent (``m²``); any other superscript digit is a footnote
    marker and is removed (its number is returned) unless ``keep_superscripts`` is set
    (footnote lines start with their own number). Where a marker is removed, the words on
    either side are separated by one space, and a space left before punctuation that
    followed the marker ("unsuitable ¹, a") is removed.
    """
    main_size = max(s.size for s in spans)
    parts: list[str] = []
    markers: list[int] = []
    prev_x1: float | None = None
    marker_gap_x1: float | None = None  # end of the text before a removed footnote marker
    for span in spans:
        text = span.text
        if not text.strip():
            continue
        if is_bullet_glyph(span):
            if parts and parts[-1] == "• ":
                continue  # the same bullet as a drawn glyph and as a text character
            if parts and not parts[-1].endswith(" "):
                parts.append(" ")
            parts.append("• ")
            prev_x1 = span.x1
            continue
        if span.size < 0.8 * main_size and text.strip().isdigit() and not keep_superscripts:
            digits = text.strip()
            if parts and parts[-1].endswith("m") and digits in SUPERSCRIPTS:
                parts.append(SUPERSCRIPTS[digits])
            else:
                markers.append(int(digits))
                if marker_gap_x1 is None:
                    marker_gap_x1 = prev_x1
            prev_x1 = span.x1
            continue
        if marker_gap_x1 is not None and parts:
            if text.lstrip()[:1] in (",", ".", ";", ":", ")"):
                parts[-1] = parts[-1].rstrip()
                text = text.lstrip()
            elif span.x0 - marker_gap_x1 > 1.0 and not parts[-1].endswith(" ") and not text.startswith(" "):
                parts.append(" ")
            marker_gap_x1 = None
        elif prev_x1 is not None and span.x0 - prev_x1 > 1.0 and parts and not parts[-1].endswith(" ") \
                and not text.startswith(" "):
            parts.append(" ")
        parts.append(text)
        prev_x1 = span.x1
    return normalise_text("".join(parts)), markers


def make_lines(spans: list[Span], keep_superscripts: bool = False) -> list[TLine]:
    """Group spans into visual lines by vertical centre (handles bullets and superscripts,
    whose boxes are offset from the text they belong to), ordered top to bottom."""
    ordered = sorted((s for s in spans if s.text.strip()), key=lambda s: (s.page, round(_centre(s), 2), s.x0))
    groups: list[list[Span]] = []
    last_centre = 0.0
    for span in ordered:
        c = _centre(span)
        if groups and groups[-1][0].page == span.page and abs(c - last_centre) <= LINE_TOLERANCE:
            groups[-1].append(span)
        else:
            groups.append([span])
        last_centre = c
    lines = []
    for group in groups:
        group.sort(key=lambda s: s.x0)
        text, markers = line_text(group, keep_superscripts)
        if text:
            lines.append(TLine(group, text, markers))
    lines.sort(key=lambda ln: (ln.page, ln.y0, ln.x0))
    return lines


def join_items(lines: list[TLine], vocab: set[str]) -> list[str]:
    """Join lines into items: a line starting with a bullet starts a new item."""
    items: list[list[str]] = []
    for ln in lines:
        if not items or ln.starts_item:
            items.append([ln.text])
        else:
            items[-1].append(ln.text)
    return [join_lines(item, vocab) for item in items]


# --------------------------------------------------------------------------- #
# Page model
# --------------------------------------------------------------------------- #


@dataclass
class Caption:
    text: str
    page: int
    y0: float
    y1: float
    x0: float
    x1: float


@dataclass
class Box:
    """A side box: a title line followed by its items."""

    title: str
    items: list[str]
    page: int
    y0: float


@dataclass
class Table:
    caption: Caption
    rows: list[str]
    notes: list[str]
    page: int


@dataclass
class Flowchart:
    caption: str | None
    columns: list[list[str]]  # box texts per reading-order column
    trailing: list[str]  # text outside any box that is not a note (e.g. a date stamp)
    notes: list[str]
    page: int
    order_note: str


@dataclass
class Item:
    """One element of the reading-order stream."""

    page: int
    key: float
    kind: str  # "line" | "box" | "table" | "flowchart"
    line: TLine | None = None
    box: Box | None = None
    table: Table | None = None
    flowchart: Flowchart | None = None


@dataclass
class ParseReport:
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def printed_page_numbers(doc: pymupdf.Document) -> dict[int, int | None]:
    """Printed page number of each PDF page, read from the footer (None if absent)."""
    numbers: dict[int, int | None] = {}
    for page in doc:
        spans = extract_spans(page, {}, {})
        footer = [s for s in spans if s.y0 >= FOOTER_TOP and s.text.strip()]
        text = " ".join(s.text.strip() for s in sorted(footer, key=lambda s: s.x0))
        match = re.fullmatch(r"(\d+)(?: HEARTS.*)?|HEARTS.*?(\d+)", text)
        numbers[page.number + 1] = int(match.group(1) or match.group(2)) if match else None
    return numbers


def horizontal_rules(page: pymupdf.Page) -> list[tuple[float, float, float]]:
    """Horizontal ruling segments ``(y, x0, x1)`` drawn on the page."""
    rules = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.height < 1.5 and rect.width > 10 and all(item[0] == "l" for item in drawing["items"]):
            rules.append((round((rect.y0 + rect.y1) / 2, 1), round(rect.x0, 1), round(rect.x1, 1)))
    return sorted(rules)


def _cluster(values: list[float], tolerance: float) -> list[float]:
    out: list[float] = []
    for v in sorted(values):
        if not out or v - out[-1] > tolerance:
            out.append(v)
    return out


def flowchart_boxes(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Rounded rectangles (drawn with curves) of a flowchart page, page background excluded."""
    boxes = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        kinds = {item[0] for item in drawing["items"]}
        if "c" in kinds and rect.width > 20 and rect.height > 8 and rect.width < page.rect.width - 20:
            boxes.append(pymupdf.Rect(rect))
    unique: dict[tuple[int, int, int, int], pymupdf.Rect] = {}
    for r in boxes:
        unique.setdefault((round(r.x0), round(r.y0), round(r.x1), round(r.y1)), r)
    return list(unique.values())


def sub_rows(cells: list[list[TLine]], gap: float = 15.0) -> list[list[list[TLine]]]:
    """Split one ruled table row into sub-rows where every column with text both above and
    below a y position has a vertical gap of more than ``gap`` there (and at least two
    columns do). Table 1's "Monogenic diabetes" row holds two such sub-rows (a and b)
    without a rule between them. Line pitch inside a cell is 11-14 pt."""
    cuts: list[float] = []
    candidates = sorted({(a.y0 + b.y0) / 2 for col in cells for a, b in zip(col, col[1:]) if b.y0 - a.y0 > gap})
    for y in candidates:
        spanning = [col for col in cells if any(ln.y0 < y for ln in col) and any(ln.y0 > y for ln in col)]
        if len(spanning) >= 2 and all(
            min(ln.y0 for ln in col if ln.y0 > y) - max(ln.y0 for ln in col if ln.y0 < y) > gap for col in spanning
        ):
            cuts.append(y)
    bounds = [-1e9, *cuts, 1e9]
    return [[[ln for ln in col if lo < ln.y0 < hi] for col in cells] for lo, hi in zip(bounds, bounds[1:])]


def reading_order(boxes: list[tuple[float, float, str]]) -> list[str]:
    """Box texts ``(y0, x0, text)`` top to bottom; boxes whose tops are within 4 pt form one
    row and are read left to right."""
    rows: list[list[tuple[float, float, str]]] = []
    for box in sorted(boxes):
        if rows and box[0] - rows[-1][0][0] <= 4.0:
            rows[-1].append(box)
        else:
            rows.append([box])
    return [text for row in rows for _, _, text in sorted(row, key=lambda b: b[1])]


def split_cells(spans: list[Span], min_gap: float = 8.0) -> list[list[Span]]:
    """Split spans into side-by-side cells separated by a vertical gutter of at least ``min_gap``."""
    groups: list[list[Span]] = []
    right = -1e9
    for span in sorted(spans, key=lambda s: s.x0):
        if groups and span.x0 - right < min_gap:
            groups[-1].append(span)
        else:
            groups.append([span])
        right = max(right, span.x1) if len(groups[-1]) > 1 else span.x1
    return groups


class Document:
    """Reads the PDF into a reading-order stream of :class:`Item` objects."""

    def __init__(self, pdf_path: Path) -> None:
        self.doc = pymupdf.open(pdf_path)
        self.report = ParseReport()
        self.footnotes: dict[tuple[int, int], str] = {}  # (page, number) -> footnote text
        self.used_footnotes: set[tuple[int, int]] = set()
        self.page_spans: dict[int, list[Span]] = {}
        for page in self.doc:
            n = page.number + 1
            if n < FIRST_CONTENT_PAGE:
                continue
            # extract_spans raises on any Private Use Area glyph; the original has none.
            self.page_spans[n] = [s for s in clean_spans(extract_spans(page, {}, {})) if s.y0 < FOOTER_TOP]
        self.vocab = build_vocabulary(ln.text for n in self.page_spans for ln in make_lines(self.page_spans[n]))
        self.report.skipped.append(
            f"PDF pp. 1-{FIRST_CONTENT_PAGE - 1}: cover, copyright, contents, acknowledgements, abbreviations"
        )

    # -- captions, boxes, footnotes ---------------------------------------- #

    def captions(self, spans: list[Span]) -> list[Caption]:
        """Caption blocks: BOX_GREY bold lines starting with "Fig." or "Table", plus continuations."""
        caps: list[list[Span]] = []
        for span in sorted(spans, key=lambda s: (s.y0, s.x0)):
            if span.text.strip().startswith(("Fig.", "Table ")):
                caps.append([span])
                continue
            best = None
            for cap in caps:
                last = max(cap, key=lambda s: s.y1)
                overlap = min(span.x1, max(s.x1 for s in cap)) - max(span.x0, min(s.x0 for s in cap))
                if overlap > 0 and 0 <= span.y0 - last.y0 <= 16:
                    best = cap
            if best is None:
                raise ValueError(f"caption continuation without caption on p. {span.page}: {span.text!r}")
            best.append(span)
        out = []
        for cap in caps:
            text = join_lines([ln.text for ln in make_lines(cap)], self.vocab)
            out.append(Caption(text, cap[0].page, min(s.y0 for s in cap), max(s.y1 for s in cap),
                               min(s.x0 for s in cap), max(s.x1 for s in cap)))
        return out

    def boxes(self, spans: list[Span]) -> list[Box]:
        """Side boxes: a title (medium weight, possibly two lines) followed by items."""
        boxes: list[Box] = []
        title_lines: list[str] = []
        for ln in make_lines(spans):
            if "Medium" in ln.main.font:
                if title_lines and boxes and not boxes[-1].items:
                    boxes[-1].title = join_lines([boxes[-1].title, ln.text], self.vocab)
                    continue
                boxes.append(Box(ln.text, [], ln.page, ln.y0))
                title_lines = [ln.text]
                continue
            if not boxes:
                raise ValueError(f"box text without a box title on p. {ln.page}: {ln.text!r}")
            box = boxes[-1]
            if ln.starts_item or not box.items or ln.y0 - prev_y > PARAGRAPH_GAP:
                box.items.append(ln.text)
            else:
                box.items[-1] = join_lines([box.items[-1], ln.text], self.vocab)
            prev_y = ln.y0
        return boxes

    def read_footnotes(self, spans: list[Span], page: int) -> list[tuple[str, TLine]]:
        """Group GREY small lines into notes. Returns ``(kind, first line)`` pairs and stores
        numbered page footnotes in ``self.footnotes``; "*"-notes are returned for tables."""
        notes: list[tuple[str, list[str], TLine]] = []
        for ln in make_lines(spans, keep_superscripts=True):
            numbered = re.match(r"^(\d+) ", ln.text) and ln.spans[0].size < 7
            if numbered or ln.text.startswith("*") or not notes:
                notes.append(("page" if numbered else "table", [ln.text], ln))
            else:
                notes[-1][1].append(ln.text)
        out = []
        for kind, texts, first in notes:
            text = join_lines(texts, self.vocab)
            if kind == "page":
                number = int(text.split(" ", 1)[0])
                self.footnotes[(page, number)] = text
            out.append((kind, first if kind == "page" else TLine(first.spans, text, [])))
        return out

    # -- tables ------------------------------------------------------------- #

    def table(self, caption: Caption, body: list[Span], page: pymupdf.Page) -> tuple[Table, set[int]]:
        """Read the table under ``caption``; returns it and the ids of the body spans it used."""
        lines = [ln for ln in make_lines(body) if ln.y0 > caption.y1]
        table_lines: list[TLine] = []
        for ln in lines:
            if ln.main.size > 9.6 or ln.main.color not in (BLACK, WHITE):
                break
            table_lines.append(ln)
        if not table_lines:
            raise ValueError(f"no table body under {caption.text!r}")
        bottom = max(ln.y1 for ln in table_lines)
        rules = [r for r in horizontal_rules(page) if caption.y1 - 1 <= r[0] <= bottom + 12]
        ys = _cluster([r[0] for r in rules], 2.0)
        xs = _cluster([x for r in rules for x in (r[1], r[2])], 3.0)
        if len(ys) < 2 or len(xs) < 2 or ys[0] > table_lines[0].y0 or ys[-1] < bottom - 1:
            raise ValueError(f"ruling lines do not enclose {caption.text!r}: ys={ys}, xs={xs}")
        cells: dict[tuple[int, int], list[Span]] = {}
        used: set[int] = set()
        for ln in table_lines:
            for span in ln.spans:
                cy, cx = _centre(span), (span.x0 + span.x1) / 2
                row = next(i for i in range(len(ys) - 1) if cy < ys[i + 1])
                col = next((i for i in range(len(xs) - 1) if cx < xs[i + 1]), len(xs) - 2)
                cells.setdefault((row, col), []).append(span)
                used.add(id(span))
        rows = [caption.text]
        for r in range(len(ys) - 1):
            row_cells = [make_lines(cells.get((r, c), [])) for c in range(len(xs) - 1)]
            for sub in sub_rows(row_cells):
                texts = [" ".join(join_items(lines, self.vocab)) for lines in sub]
                while texts and not texts[-1]:
                    texts.pop()
                if texts:
                    rows.append(normalise_text(" | ".join(texts)))  # an empty first cell leaves "| DKA | HHS"
        return Table(caption, rows, [], caption.page), used

    # -- flowcharts --------------------------------------------------------- #

    def flowchart(self, page: pymupdf.Page, spans: list[Span], spec: FlowchartSpec,
                  caption: Caption | None) -> Flowchart:
        rects = flowchart_boxes(page)
        in_box: dict[int, list[Span]] = {}
        outside: list[Span] = []
        for span in spans:
            cx, cy = (span.x0 + span.x1) / 2, _centre(span)
            containing = [i for i, r in enumerate(rects) if r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1]
            if containing:
                best = min(containing, key=lambda i: rects[i].width * rects[i].height)
                in_box.setdefault(best, []).append(span)
            else:
                outside.append(span)
        columns: list[list[tuple[float, float, str]]] = [[] for _ in range(len(spec.columns) + 1)]
        for i, box_spans in in_box.items():
            r = rects[i]
            col = next((k for k, bound in enumerate(spec.columns) if r.x0 < bound), len(spec.columns))
            cells = [" ".join(join_items(make_lines(cell), self.vocab)) for cell in split_cells(box_spans)]
            columns[col].append((round(r.y0, 1), round(r.x0, 1), " | ".join(cells)))
        notes: list[str] = []
        trailing: list[str] = []
        for ln in make_lines(outside):
            if ln.text.startswith(("#", "*", "†")):
                notes.append(ln.text)
            elif ln.main.size <= 8.5 and notes:
                notes[-1] = join_lines([notes[-1], ln.text], self.vocab)
            else:
                trailing.append(ln.text)
        return Flowchart(
            caption.text if caption else None,
            [reading_order(col) for col in columns if col],
            trailing,
            notes,
            page.number + 1,
            spec.order_note,
        )

    # -- stream ------------------------------------------------------------- #

    def items(self) -> list[Item]:
        """All pages as one reading-order stream."""
        out: list[Item] = []
        for page in self.doc:
            n = page.number + 1
            if n not in self.page_spans:
                continue
            out.extend(self.page_items(page, self.page_spans[n]))
        return out

    def page_items(self, page: pymupdf.Page, spans: list[Span]) -> list[Item]:
        n = page.number + 1
        caption_spans = [s for s in spans if s.color == BOX_GREY and is_bold(s)]
        box_spans = [s for s in spans if s.color == BOX_GREY and not is_bold(s)]
        note_spans = [s for s in spans if s.color == GREY and s.size <= 9.5]
        body = [s for s in spans if s not in caption_spans and s not in box_spans and s not in note_spans]
        captions = self.captions(caption_spans)
        items: list[Item] = []

        spec = FLOWCHARTS.get(n)
        if spec is not None:
            caption = next((c for c in captions if spec.caption_prefix and c.text.startswith(spec.caption_prefix)), None)
            heading = [s for s in body if s.size > 20]
            fig_spans = [s for s in body if s.size <= 20] + [s for s in note_spans if s.text.strip().startswith("*")]
            note_spans = [s for s in note_spans if s not in fig_spans]
            chart = self.flowchart(page, fig_spans, spec, caption)
            items.append(Item(n, caption.y0 if caption else max(s.y1 for s in heading), "flowchart", flowchart=chart))
            body = heading
            captions = [c for c in captions if c is not caption]

        tables: list[Table] = []
        for cap in captions:
            if cap.text.startswith("Table "):
                table, used = self.table(cap, body, page)
                body = [s for s in body if id(s) not in used]
                tables.append(table)
                items.append(Item(n, cap.y0, "table", table=table))
            else:
                self.report.skipped.append(f"p. {n}: photograph '{cap.text}' (caption only)")

        for kind, ln in self.read_footnotes(note_spans, n):
            if kind == "table":
                above = [t for t in tables if t.caption.y0 < ln.y0]
                if not above:
                    raise ValueError(f"table note without a table on p. {n}: {ln.text!r}")
                max(above, key=lambda t: t.caption.y0).notes.append(ln.text)

        lines = make_lines(body)
        for ln in lines:
            items.append(Item(n, ln.y0, "line", line=ln))
        groups = paragraph_ranges(lines)
        for box in self.boxes(box_spans):
            key = box.y0
            for y0, y1 in groups:
                if y0 <= box.y0 <= y1 + 1:
                    key = y1 + 0.01
            items.append(Item(n, key, "box", box=box))
        items.sort(key=lambda it: (it.key, {"line": 0, "box": 1, "table": 1, "flowchart": 1}[it.kind]))
        return items


def paragraph_ranges(lines: list[TLine]) -> list[tuple[float, float]]:
    """(first line y0, last line y0) of each paragraph on a page, by the paragraph rules."""
    ranges: list[list[float]] = []
    prev: TLine | None = None
    for ln in lines:
        if is_heading(ln) or prev is None or ln.starts_item or ln.y0 - prev.y0 > PARAGRAPH_GAP:
            ranges.append([ln.y0, ln.y0])
        else:
            ranges[-1][1] = ln.y0
        prev = ln
    return [(a, b) for a, b in ranges]


def heading_level(line: TLine) -> int | None:
    """Heading level by style, or None for body text."""
    s = line.main
    if s.color == BLUE and s.size >= 29:
        return 1
    if s.color == BLUE and s.size >= 14.4:
        return 2
    if s.color == BLUE and s.size >= 13.9:
        return 3
    if s.color == BLACK and abs(s.size - 14.0) < 0.2 and is_bold(s):
        return 2  # subtitle of the opening page
    if s.color == GREY and s.size >= 11.4 and is_bold(s):
        return 4
    if s.color == BLACK and abs(s.size - 12.0) < 0.2 and is_italic(s):
        return 5
    return None


def is_heading(line: TLine) -> bool:
    return heading_level(line) is not None


# --------------------------------------------------------------------------- #
# Sections and chunks
# --------------------------------------------------------------------------- #


@dataclass
class Block:
    text: str
    page_start: int
    page_end: int
    markers: list[tuple[int, int]] = field(default_factory=list)  # (page, footnote number)

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Leaf:
    path: list[str]
    blocks: list[Block] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(b.words for b in self.blocks)


@dataclass
class Chunk:
    chunk_id: str
    section_path: list[str]
    chunk_type: str
    source_text: str
    page_start: int
    page_end: int
    notes: list[str] = field(default_factory=list)
    context_header: str | None = None
    source_discrepancy: str | None = None
    printed_page_start: int | None = None
    printed_page_end: int | None = None
    flow_text: str | None = None  # derived, flowcharts only (see hearts_d_derived.py)

    @property
    def text(self) -> str:
        """context_header + (flow_text or source_text) + footnotes, with typo corrections."""
        return compose_corrected_text(self.flow_text or self.source_text, self.context_header, self.notes)[0]

    @property
    def text_corrections(self) -> list[dict[str, object]]:
        return compose_corrected_text(self.flow_text or self.source_text, self.context_header, self.notes)[1]

    @property
    def words(self) -> int:
        return len(self.text.split())

    def to_record(self) -> dict[str, object]:
        return build_record(
            chunk_id=self.chunk_id,
            collection=COLLECTION,
            doc=DOC_TITLE,
            doc_id=DOC_ID,
            section_path=self.section_path,
            context_header=self.context_header,
            chunk_type=self.chunk_type,
            index_default=True,
            source_discrepancy=self.source_discrepancy,
            source_text=self.source_text,
            flow_text=self.flow_text,
            text=self.text,
            text_corrections=self.text_corrections,
            pdf_page_start=self.page_start,
            pdf_page_end=self.page_end,
            printed_page_start=self.printed_page_start,
            printed_page_end=self.printed_page_end,
            source_url=SOURCE_URL,
            licence=LICENCE,
        )


def context_header(path: list[str]) -> str:
    return " - ".join([HEADER_PREFIX, *path]) + ":"


def slugify(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:max_words])


def chapter_slug(title: str) -> str:
    """"1 Clinical ..." -> "ch1"; "Annex 2: ..." -> "annex2"; "Introduction" -> "intro"."""
    if m := re.match(r"^(\d+) ", title):
        return f"ch{m.group(1)}"
    if m := re.match(r"^Annex (\d+)", title):
        return f"annex{m.group(1)}"
    if title.startswith("Introduction"):
        return "intro"
    return "about"


class Walker:
    """Consumes the item stream: tracks the heading path, builds prose leaves and emits
    table and flowchart chunks."""

    def __init__(self, document: Document) -> None:
        self.document = document
        self.vocab = document.vocab
        self.headings: dict[int, str] = {}
        self.leaves: list[Leaf] = []
        self.figures: list[Chunk] = []
        self.skip_level: int | None = None
        self._lines: list[TLine] = []
        self._pending_heading: list[TLine] = []

    @property
    def path(self) -> list[str]:
        return [self.headings[k] for k in sorted(self.headings)]

    def run(self, items: list[Item]) -> None:
        for item in items:
            if item.kind == "line":
                assert item.line is not None
                self.line(item.line)
                continue
            self.flush_heading()
            if self.skip_level is not None:
                continue
            if item.kind == "box":
                assert item.box is not None
                self.flush_paragraph()
                box = item.box
                self.add_block(Block("\n".join([box.title, *box.items]), box.page, box.page))
            elif item.kind == "table":
                assert item.table is not None
                self.flush_paragraph()
                self.emit_table(item.table)
            elif item.kind == "flowchart":
                assert item.flowchart is not None
                self.flush_paragraph()
                self.emit_flowchart(item.flowchart)
        self.flush_heading()
        self.flush_paragraph()

    # headings ------------------------------------------------------------- #

    def line(self, ln: TLine) -> None:
        level = heading_level(ln)
        if level is not None:
            if self._pending_heading and heading_level(self._pending_heading[0]) == level \
                    and ln.page == self._pending_heading[-1].page and ln.y0 - self._pending_heading[-1].y0 < 40:
                self._pending_heading.append(ln)
                return
            self.flush_heading()
            self.flush_paragraph()
            self._pending_heading = [ln]
            return
        self.flush_heading()
        if self.skip_level is not None:
            return
        self.paragraph_line(ln)

    def flush_heading(self) -> None:
        if not self._pending_heading:
            return
        lines = self._pending_heading
        self._pending_heading = []
        level = heading_level(lines[0])
        assert level is not None
        title = join_lines([ln.text for ln in lines], self.vocab)
        if self.skip_level is not None and level <= self.skip_level:
            self.skip_level = None
        if self.skip_level is not None:
            return
        if title in SKIPPED_HEADINGS:
            self.skip_level = level
            self.document.report.skipped.append(f"p. {lines[0].page}: section '{title}' ({SKIPPED_HEADINGS[title]})")
            return
        self.headings = {k: v for k, v in self.headings.items() if k < level}
        self.headings[level] = title
        self.leaves.append(Leaf(self.path))

    # paragraphs ------------------------------------------------------------ #

    def paragraph_line(self, ln: TLine) -> None:
        if self._lines:
            prev = self._lines[-1]
            same_page_break = ln.page == prev.page and ln.y0 - prev.y0 > PARAGRAPH_GAP
            page_break = ln.page != prev.page and re.search(r"[.:]$", prev.text) is not None
            if ln.starts_item or same_page_break or page_break:
                self.flush_paragraph()
        self._lines.append(ln)

    def flush_paragraph(self) -> None:
        if not self._lines:
            return
        lines = self._lines
        self._lines = []
        markers = [(ln.page, m) for ln in lines for m in ln.markers]
        text = join_lines([ln.text for ln in lines], self.vocab)
        self.add_block(Block(text, lines[0].page, lines[-1].page, markers))

    def add_block(self, block: Block) -> None:
        if not self.leaves or self.leaves[-1].path != self.path:
            self.leaves.append(Leaf(self.path))
        self.leaves[-1].blocks.append(block)

    # tables and flowcharts --------------------------------------------------- #

    def emit_table(self, table: Table) -> None:
        number = re.match(r"Table (\d+)", table.caption.text)
        assert number is not None
        self.figures.append(Chunk(
            chunk_id=f"{DOC_ID}-table-{number.group(1)}",
            section_path=[*self.path, table.caption.text],
            chunk_type="table",
            source_text="\n".join(table.rows),
            page_start=table.page,
            page_end=table.page,
            notes=list(table.notes),
            context_header=context_header(self.path),
        ))

    def emit_flowchart(self, chart: Flowchart) -> None:
        if chart.caption:
            number = re.match(r"Fig\. (\d+)", chart.caption)
            assert number is not None
            base_id = f"{DOC_ID}-fig-{number.group(1)}"
            path = [self.path[0], chart.caption]
        else:
            base_id = f"{DOC_ID}-{chapter_slug(self.path[0])}-flowchart"
            path = self.path[:1]
        # A flowchart fills its own page, so only the chapter it is printed in is used as context.
        header = context_header(path[:1]) + f" flowchart, boxes listed {chart.order_note} (arrows not reproduced)"
        head = [chart.caption] if chart.caption else []
        whole = [*head, *[t for col in chart.columns for t in col], *chart.trailing]
        parts = [whole]
        if len(" ".join(whole).split()) + len(" ".join(chart.notes).split()) > MAX_WORDS:
            parts = [[*head, *col] for col in chart.columns]
            parts[-1].extend(chart.trailing)
        for i, lines in enumerate(parts, 1):
            text = "\n".join(lines)
            notes = [n for n in chart.notes if marker_in(text, n.split(" ", 1)[0])] if len(parts) > 1 else chart.notes
            self.figures.append(Chunk(
                chunk_id=base_id if len(parts) == 1 else f"{base_id}-part{i}",
                section_path=path,
                chunk_type="flowchart",
                source_text=text,
                page_start=chart.page,
                page_end=chart.page,
                notes=list(notes),
                context_header=header + (f", part {i} of {len(parts)}" if len(parts) > 1 else ""),
            ))


def marker_in(text: str, marker: str) -> bool:
    """True if the note marker (e.g. "*", "**", "#", "†") occurs in ``text`` as a whole marker."""
    ch = re.escape(marker[0])
    return re.search(rf"(?<!{ch}){re.escape(marker)}(?!{ch})", text) is not None


def leaves_to_chunks(walker: Walker) -> list[Chunk]:
    """Merge small neighbouring sections, split large ones, and build prose chunks."""
    leaves = [leaf for leaf in walker.leaves if leaf.blocks]
    groups: list[list[Leaf]] = []
    for leaf in leaves:
        if groups:
            prev = groups[-1]
            last = prev[-1].path
            # Only siblings or parent and child sections are merged, never distant sections.
            related = len(common_prefix(last, leaf.path)) >= min(len(last), len(leaf.path)) - 1
            prev_words = sum(lf.words for lf in prev)
            if related and prev[0].path[:1] == leaf.path[:1] and (prev_words < MIN_WORDS or leaf.words < MIN_WORDS) \
                    and prev_words + leaf.words <= MAX_WORDS:
                prev.append(leaf)
                continue
        groups.append([leaf])

    chunks: list[Chunk] = []
    for group in groups:
        prefix = common_prefix_all([lf.path for lf in group])
        blocks: list[Block] = []
        prev_path = prefix
        for leaf in group:
            # Headings below the chunk's own section_path are kept in the text (they are source text).
            extra = leaf.path[max(len(prefix), len(common_prefix(prev_path, leaf.path))):]
            if extra:
                page = leaf.blocks[0].page_start
                blocks.append(Block("\n".join(extra), page, page))
            blocks.extend(leaf.blocks)
            prev_path = leaf.path
        for part in split_blocks(blocks):
            chunks.append(Chunk(
                chunk_id="",
                section_path=prefix,
                chunk_type="prose",
                source_text="\n".join(b.text for b in part),
                page_start=part[0].page_start,
                page_end=part[-1].page_end,
                notes=footnotes_for(part, walker.document),
            ))
            chunks[-1].context_header = context_header(chunks[-1].section_path)
            chunks[-1].chunk_id = prose_id(chunks[-1].section_path)
    return chunks


def common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def common_prefix_all(paths: list[list[str]]) -> list[str]:
    prefix = paths[0]
    for p in paths[1:]:
        prefix = common_prefix(prefix, p)
    return prefix


def split_blocks(blocks: list[Block]) -> list[list[Block]]:
    """Split a block list into parts of at most MAX_WORDS, as evenly as block boundaries allow."""
    total = sum(b.words for b in blocks)
    if total <= MAX_WORDS:
        return [blocks]
    n_parts = -(-total // MAX_WORDS)
    target = total / n_parts
    parts: list[list[Block]] = [[]]
    words = 0
    for block in blocks:
        if parts[-1] and (words + block.words > MAX_WORDS or words >= target):
            parts.append([])
            words = 0
        parts[-1].append(block)
        words += block.words
    return parts


def footnotes_for(blocks: list[Block], document: Document) -> list[str]:
    notes = []
    for block in blocks:
        for key in block.markers:
            if key not in document.footnotes:
                raise ValueError(f"footnote marker {key[1]} on p. {key[0]} has no footnote")
            document.used_footnotes.add(key)
            if document.footnotes[key] not in notes:
                notes.append(document.footnotes[key])
    return notes


def prose_id(path: list[str]) -> str:
    chapter = chapter_slug(path[0])
    tail = slugify(path[-1]) if len(path) > 1 else ""
    return f"{DOC_ID}-{chapter}" + (f"-{tail}" if tail else "")


def make_ids_unique(chunks: list[Chunk]) -> None:
    """Append -p1, -p2 ... to ids that occur more than once (in document order)."""
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.chunk_id] = counts.get(c.chunk_id, 0) + 1
    seen: dict[str, int] = {}
    for c in chunks:
        if counts[c.chunk_id] > 1:
            seen[c.chunk_id] = seen.get(c.chunk_id, 0) + 1
            c.chunk_id = f"{c.chunk_id}-p{seen[c.chunk_id]}"


def assign_discrepancies(chunks: list[Chunk]) -> dict[str, list[str]]:
    """Attach each discrepancy to every chunk containing one of its anchors.

    Raises if an anchor is found in no chunk, so that a discrepancy is never silently lost.
    Returns ``{discrepancy key: [chunk ids]}``.
    """
    where: dict[str, list[str]] = {}
    for d in DISCREPANCIES:
        for anchor in d.anchors:
            hits = [c for c in chunks if normalise_for_match(anchor) in normalise_for_match(c.source_text)]
            if not hits:
                raise ValueError(f"discrepancy {d.key}: anchor {anchor!r} not found in any chunk")
            for c in hits:
                existing = c.source_discrepancy.split("\n") if c.source_discrepancy else []
                if d.description not in existing:
                    c.source_discrepancy = "\n".join([*existing, d.description])
                    where.setdefault(d.key, []).append(c.chunk_id)
    return where


def parse(pdf_path: Path | None = None) -> tuple[list[Chunk], Document, dict[str, list[str]]]:
    """Parse the PDF; returns chunks in document order, the document and the discrepancy map."""
    pdf_path = pdf_path or get_source("hearts_d_pdf").path
    document = Document(pdf_path)
    walker = Walker(document)
    walker.run(document.items())
    prose = leaves_to_chunks(walker)
    unused = set(document.footnotes) - document.used_footnotes
    if unused:
        raise ValueError(f"footnotes not attached to any chunk: {sorted(unused)}")
    chunks = sorted(prose + walker.figures, key=lambda c: (c.page_start, 0 if c.chunk_type == "prose" else 1))
    make_ids_unique(chunks)
    printed = printed_page_numbers(document.doc)
    for c in chunks:
        c.printed_page_start = printed.get(c.page_start)
        c.printed_page_end = printed.get(c.page_end)
    where = assign_discrepancies(chunks)
    assign_flow_text(chunks)
    return chunks, document, where


def assign_flow_text(chunks: list[Chunk]) -> None:
    """Attach the hand-transcribed ``flow_text`` to every flowchart chunk (and only to those)."""
    keys = set(FLOW_TEXT)
    for c in chunks:
        if c.chunk_type != "flowchart":
            continue
        key = c.chunk_id.removeprefix(f"{DOC_ID}-")
        if key not in FLOW_TEXT:
            raise ValueError(f"no flow_text for flowchart chunk {c.chunk_id}")
        c.flow_text = "\n".join(FLOW_TEXT[key])
        keys.discard(key)
    if keys:
        raise ValueError(f"flow_text for unknown chunks: {sorted(keys)}")


def statement_page(document: Document, chunk: Chunk, anchor: str) -> int:
    """The PDF page within the chunk's range on which ``anchor`` is printed."""
    if chunk.page_start == chunk.page_end:
        return chunk.page_start  # e.g. flowcharts, whose page text interleaves the boxes
    needle = normalise_for_match(anchor)
    for page in range(chunk.page_start, chunk.page_end + 1):
        lines = [ln.text for ln in make_lines(document.page_spans[page])]
        if needle in normalise_for_match(join_lines(lines, document.vocab)):
            return page
    raise ValueError(f"dose statement {anchor!r} not found on the pages of {chunk.chunk_id}")


def dose_rows(chunks: list[Chunk], document: Document) -> list[dict[str, object]]:
    """Rows of ``hearts_d_doses.csv`` (see ``hearts_d_derived.DOSE_STATEMENTS``).

    Each statement's anchor must occur in exactly one chunk; its chunk id, PDF page and
    printed page are taken from there. Raises if an anchor is missing or ambiguous.
    """
    printed = printed_page_numbers(document.doc)
    rows = []
    for st in DOSE_STATEMENTS:
        hits = [c for c in chunks if normalise_for_match(st.anchor) in normalise_for_match(c.source_text)]
        if len(hits) != 1:
            raise ValueError(f"dose statement {st.anchor!r} found in {len(hits)} chunks")
        chunk = hits[0]
        page = statement_page(document, chunk, st.anchor)
        rows.append({
            "drug": st.drug, "drug_as_printed": st.drug_as_printed, "drug_source": st.drug_source,
            "dose": st.dose, "dose_corrected": apply_corrections(st.dose)[0], "dose_basis": st.dose_basis,
            "route": st.route, "frequency": st.frequency, "timing": st.timing,
            "dosing_as_printed": st.dosing_as_printed, "figure_or_section": st.figure_or_section, "step": st.step, "branch": st.branch, "condition": st.condition, "chunk_id": chunk.chunk_id,
            "pdf_page": page, "printed_page": printed.get(page) if printed.get(page) is not None else "",
            "source_discrepancy": "; ".join(st.discrepancies),
        })
    return rows


def write_doses(rows: list[dict[str, object]], path: Path = DOSES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DOSE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(chunks: list[Chunk], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_record(), ensure_ascii=False) + "\n")


def print_summary(chunks: list[Chunk], document: Document, where: dict[str, list[str]]) -> None:
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.chunk_type] = by_type.get(c.chunk_type, 0) + 1
    words = [c.words for c in chunks]
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[2])}: {by_type}")
    print(f"Words per chunk (text): min {min(words)}, max {max(words)}, mean {sum(words) / len(words):.0f}")
    print(f"Footnotes attached: {len(document.used_footnotes)}")
    print("Skipped:")
    for s in document.report.skipped:
        print(f"  - {s}")
    print("Source discrepancies:")
    for key, ids in where.items():
        print(f"  - {key}: {', '.join(ids)}")


def main() -> None:
    chunks, document, where = parse()
    write_jsonl(chunks)
    rows = dose_rows(chunks, document)
    write_doses(rows)
    print_summary(chunks, document, where)
    print(f"Wrote {len(rows)} dose statements to {DOSES_PATH.relative_to(DOSES_PATH.parents[2])}")
    for c in chunks:
        if c.text_corrections:
            print(f"  text_corrections {c.chunk_id}: {c.text_corrections}")


if __name__ == "__main__":
    sys.exit(main())
