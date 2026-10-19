"""Inspect the raw WHO sources and write ``notes/source_inspection.md``.

This script does not produce chunks. It measures the structure of each source
(spine, headings, tables, images, page ranges, extraction quality) so that the
choice of primary source and parsing strategy is based on evidence. Every number
in the report is computed here; the report can be regenerated at any time with::

    python scripts/inspect_sources.py
"""

from __future__ import annotations

import html
import re
import sys
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pdfplumber
import pymupdf
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.parse.pdf_utils import PUA_RE, type3_pua_glyph_hashes  # noqa: E402
from src.parse.sources import get_source  # noqa: E402

REPORT_PATH = ROOT / "notes" / "source_inspection.md"
LOWER_UTI_TITLE = "Lower urinary tract infection"

# Hypertension PDF styling, measured from the file (see the report for evidence).
HTN_RED = 0xB6294C
HTN_BODY = 0x221F21
HTN_WHITE = 0xFDFEFF
HTN_GREY = 0x6B6E71
GRADING_RE = re.compile(r"^(Strong|Conditional) recommendation,\s*(.+?)\s*evidence\s*$")

warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def md_table(header: list[str], rows: list[list[object]]) -> str:
    """Render a Markdown table, escaping pipes in cell text."""

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def css_class(tag: Tag) -> str:
    """Return a tag's class attribute as a string (lxml-xml gives str, html parsers give lists)."""
    value = tag.get("class", "")
    return " ".join(value) if isinstance(value, list) else str(value)


# --------------------------------------------------------------------------- #
# AWaRe EPUB
# --------------------------------------------------------------------------- #
@dataclass
class EpubDoc:
    index: int  # 1-based spine position
    href: str
    soup: BeautifulSoup
    raw: str

    @property
    def name(self) -> str:
        return PurePosixPath(self.href).name

    @property
    def h1(self) -> str:
        tag = self.soup.find("h1")
        return squash(tag.get_text(" ")) if tag else ""


def load_epub_spine(path: Path) -> list[EpubDoc]:
    """Read the OPF spine and parse every spine document (zipfile + BeautifulSoup)."""
    with zipfile.ZipFile(path) as zf:
        container = BeautifulSoup(zf.read("META-INF/container.xml"), "lxml-xml")
        opf_path = container.find("rootfile")["full-path"]
        opf_dir = PurePosixPath(opf_path).parent
        opf = BeautifulSoup(zf.read(opf_path), "lxml-xml")
        manifest = {item["id"]: item["href"] for item in opf.find_all("item")}
        docs = []
        for i, ref in enumerate(opf.find("spine").find_all("itemref"), start=1):
            href = str(opf_dir / manifest[ref["idref"]])
            raw = zf.read(href).decode("utf-8")
            docs.append(EpubDoc(i, href, BeautifulSoup(raw, "lxml-xml"), raw))
    return docs


def primary_care_slice(docs: list[EpubDoc]) -> list[EpubDoc]:
    """Spine documents strictly between the PRIMARY HEALTH CARE and HOSPITAL FACILITY dividers."""
    names = [d.name for d in docs]
    start = names.index("Primary_Health_Care.xhtml")
    end = names.index("HOSPITAL_FACILITY.xhtml")
    return docs[start + 1 : end]


def classify_image(src: str) -> str:
    name = PurePosixPath(src).name.lower()
    if "icon" in name:
        return "box icon (Note / Key messages / Important ...)"
    if name.startswith("00_"):
        return "population badge (adults / children)"
    return "content image (infographic / figure)"


def inspect_epub(lines: list[str]) -> dict[str, object]:
    src = get_source("aware_epub")
    docs = load_epub_spine(src.path)
    pc = primary_care_slice(docs)
    pc_names = {d.name for d in pc}
    facts: dict[str, object] = {"pc_names": pc_names, "docs": docs}

    lines.append("## 1. AWaRe antibiotic book – EPUB\n")
    lines.append(f"File: `data/raw/{src.filename}` (ISBN {src.isbn}, ePub version).\n")

    # --- spine ------------------------------------------------------------- #
    lines.append("### 1.1 Spine and chapter mapping\n")
    lines.append(
        f"The OPF spine lists **{len(docs)} XHTML documents**. Each numbered chapter is its own "
        "document (`chapterNN.xhtml`); part dividers (`Primary_Health_Care.xhtml`, "
        "`HOSPITAL_FACILITY.xhtml`, `RESERVE_ANTIBIOTICS.xhtml`, `DOSING_GUIDANCE.xhtml`) contain a "
        "single image and no text.\n"
    )
    rows = []
    for d in docs:
        part = "**primary care**" if d.name in pc_names else ""
        rows.append(
            [d.index, f"`{d.name}`", d.h1 or "(no h1)", len(d.soup.find_all("table")), len(d.soup.find_all("img")), part]
        )
    lines.append(md_table(["#", "Spine document", "First h1", "tables", "imgs", "Scope"], rows))
    lines.append(
        f"\nPrimary health care chapters = spine items {pc[0].index}–{pc[-1].index} "
        f"(`{pc[0].name}` *{pc[0].h1}* to `{pc[-1].name}` *{pc[-1].h1}*), **{len(pc)} chapters**.\n"
    )

    # --- headings ------------------------------------------------------------ #
    lines.append("### 1.2 Heading structure\n")
    tag_class = Counter()
    for d in docs:
        for t in d.soup.find_all(re.compile(r"^h[1-6]$")):
            tag_class[(t.name, css_class(t) or "-")] += 1
    rows = [[t, c, n] for (t, c), n in sorted(tag_class.items())]
    lines.append("Heading tags across the whole book:\n")
    lines.append(md_table(["tag", "class", "count"], rows))

    h_counts = Counter()
    h2_texts = Counter()
    h3_texts = Counter()
    for d in pc:
        for t in d.soup.find_all(re.compile(r"^h[1-6]$")):
            h_counts[t.name] += 1
            if t.name == "h2":
                h2_texts[squash(t.get_text(" "))] += 1
            elif t.name == "h3":
                h3_texts[squash(t.get_text(" "))] += 1
    lines.append(
        "\nWithin the primary care chapters: "
        + ", ".join(f"`{k}` × {v}" for k, v in sorted(h_counts.items()))
        + ". No `h4`–`h6` are used.\n"
    )
    lines.append(
        "- **Chapter** = `h1` (exactly one per chapter document, e.g. *23. Lower urinary tract infection*).\n"
        "- **Section** = `h2` (classes `h2a`/`h2b` are styling variants only). Most common section headings:\n"
    )
    lines.append(md_table(["h2 text", "chapters"], [[k, v] for k, v in h2_texts.most_common(15)]))
    lines.append("\n- **Subsection** = `h3` (class `h3a`). Most common:\n")
    lines.append(md_table(["h3 text", "count"], [[k, v] for k, v in h3_texts.most_common(10)]))
    other_titles = Counter()
    for d in pc:
        for p in d.soup.find_all("p", class_=["box_tit", "Infographic-Title-1-A", "table_titb"]):
            other_titles["p." + css_class(p)] += 1
        for c in d.soup.find_all("caption"):
            other_titles["caption." + css_class(c)] += 1
    lines.append(
        "\nOther title-like elements that are *not* h-tags but matter for chunking: "
        + ", ".join(f"`{k}` × {v}" for k, v in sorted(other_titles.items()))
        + ". Boxed callouts (*Note*, *Key messages*, *Important*) use `p.box_tit`; table titles are "
        "`<caption>` elements.\n"
    )

    # --- tables vs images ---------------------------------------------------- #
    lines.append("### 1.3 Tables vs images\n")
    all_tables = sum(len(d.soup.find_all("table")) for d in docs)
    all_imgs = sum(len(d.soup.find_all("img")) for d in docs)
    pc_tables = sum(len(d.soup.find_all("table")) for d in pc)
    pc_imgs = sum(len(d.soup.find_all("img")) for d in pc)
    img_kinds_all = Counter(classify_image(i["src"]) for d in docs for i in d.soup.find_all("img"))
    img_kinds_pc = Counter(classify_image(i["src"]) for d in pc for i in d.soup.find_all("img"))
    rows = [
        ["`<table>`", all_tables, pc_tables],
        ["`<img>` (all)", all_imgs, pc_imgs],
    ]
    for kind in sorted(set(img_kinds_all) | set(img_kinds_pc)):
        rows.append([f"&nbsp;&nbsp;↳ {kind}", img_kinds_all[kind], img_kinds_pc[kind]])
    lines.append(md_table(["element", "whole book", "primary care chapters"], rows))
    alts = Counter(i.get("alt", "") for d in pc for i in d.soup.find_all("img") if classify_image(i["src"]).startswith("content"))
    lines.append(
        f"\nContent images in primary care carry no useful alt text (alt values: {dict(alts)}). "
        "They are the one-page clinical *infographic* summaries (`NN_<Topic>_a.jpg` ...); their text "
        "is not recoverable from the EPUB without OCR.\n"
    )
    aware_classes = Counter()
    for d in pc:
        for s in d.soup.find_all("span", class_=re.compile(r"^back_")):
            aware_classes[css_class(s)] += 1
    lines.append(
        "AWaRe category of each antibiotic is encoded only as a CSS class on a `<span>` around the drug "
        f"name, counts in primary care: {dict(aware_classes)}. Every table footer carries the legend *ACCESS "
        "antibiotics are indicated in green, WATCH antibiotics in orange and RESERVE antibiotics in red*, so "
        "`back_verde` (Portuguese *green*) = ACCESS and `back_amarelo` (*yellow*, rendered orange) = WATCH; "
        "no RESERVE class occurs in the primary care chapters.\n"
    )
    pagebreaks = [
        int(m) for d in pc for m in re.findall(r'epub:type="pagebreak"[^>]*title="(\d+)"', d.raw)
    ]
    lines.append(
        f"Printed page numbers are preserved as `epub:type=\"pagebreak\"` markers: {len(pagebreaks)} markers "
        f"in primary care, pages {min(pagebreaks)}–{max(pagebreaks)}. Chunks can therefore cite the same page "
        "numbers as the printed/PDF book.\n"
    )

    # --- lower UTI ------------------------------------------------------------ #
    uti = next(d for d in pc if LOWER_UTI_TITLE.lower() in d.h1.lower())
    lines.append(f"### 1.4 *{uti.h1}* (`{uti.name}`): how dosing is represented\n")
    tables = uti.soup.find_all("table")
    rows = []
    dosing_table: Tag | None = None
    for t in tables:
        cap = squash(t.find("caption").get_text(" ")) if t.find("caption") else "(no caption)"
        rows.append([cap, len(t.find_all("tr"))])
        if "antibiotic treatment" in cap.lower():
            dosing_table = t
    lines.append(md_table(["table caption", "`<tr>` rows"], rows))
    imgs = [(i["src"], i.get("alt", "")) for i in uti.soup.find_all("img") if classify_image(i["src"]).startswith("content")]
    lines.append(
        "\nContent images in this chapter: " + ", ".join(f"`{PurePosixPath(s).name}` (alt=\"{a}\")" for s, a in imgs) + ".\n"
    )
    assert dosing_table is not None, "Lower UTI dosing table not found"
    body_rows = dosing_table.find("tbody").find_all("tr", recursive=False)
    drug_rows = [r for r in body_rows if r.find("span", class_=re.compile(r"^back_"))]
    epub_drugs = [squash(r.find("td").find("span", class_=re.compile(r"^back_")).get_text(" ")) for r in drug_rows]
    facts["epub_uti_drugs"] = epub_drugs
    n_li = len(dosing_table.find_all("li"))
    n_sup = len(dosing_table.find("tbody").find_all("sup"))
    n_foot = len(dosing_table.find("tfoot").find_all("p")) if dosing_table.find("tfoot") else 0
    headers = [squash(td.get_text(" ")) for td in body_rows[1].find_all("td")]
    lines.append(
        "**Dosing is an HTML `<table>`** (*Table 23.5 – Empiric antibiotic treatment for lower urinary tract "
        "infections*), not an image, list-only block or free paragraphs:\n\n"
        f"- `<tbody>` has {len(body_rows)} rows: one full-width *Important* callout (`colspan=\"3\"`), one header "
        f"row ({' / '.join(headers)}), and {len(drug_rows)} drug rows ({', '.join(epub_drugs)}).\n"
        "- Each drug row has 3 `<td>`: adult regimen, child regimen, total duration. Inside a cell the regimen "
        f"is a sequence of `<p class=\"box_p\">` paragraphs (weight bands are one `<p>` each) or a `<ul>` "
        f"({n_li} `<li>` in this table, e.g. nitrofurantoin formulations).\n"
        "- Drug names sit in `<span class=\"back_verde\">` (= ACCESS, green); footnote markers are `<sup>` "
        f"({n_sup} in the body) resolved by {n_foot} `<p class=\"legenda\">` notes in `<tfoot>`.\n"
        "- Quirk: `<colgroup>` declares 2 `<col>` while rows have 3 cells – harmless for parsing but shows "
        "the markup was hand-edited (Sigil).\n"
        "- The same regimens also appear inside the infographic JPEGs, which are image-only in the EPUB.\n"
    )
    nitro = next(r for r in drug_rows if "Nitrofurantoin" in r.get_text())
    snippet = re.sub(r"\n\s*\n", "\n", str(nitro)).strip()
    lines.append("Raw HTML of the nitrofurantoin row:\n\n```html\n" + snippet + "\n```\n")
    return facts


# --------------------------------------------------------------------------- #
# AWaRe PDF
# --------------------------------------------------------------------------- #
def inspect_aware_pdf(lines: list[str], epub_facts: dict[str, object]) -> None:
    src = get_source("aware_pdf")
    doc = pymupdf.open(src.path)
    lines.append("## 2. AWaRe antibiotic book – PDF\n")
    lines.append(
        f"File: `data/raw/{src.filename}` (ISBN {src.isbn}), {doc.page_count} pages, producer "
        f"*{doc.metadata.get('producer')}*. The PDF has an outline (bookmarks) and page labels that "
        "match the printed page numbers.\n"
    )
    toc = [(lvl, title.strip(), page) for lvl, title, page in doc.get_toc(simple=True) if lvl == 1]
    titles = [t for _, t, _ in toc]
    i_pc = titles.index("PRIMARY HEALTH CARE")
    i_hosp = titles.index("HOSPITAL FACILITY")
    chapters = toc[i_pc + 1 : i_hosp]
    pc_first, pc_last = toc[i_pc][2], toc[i_hosp][2] - 1

    def label(pno: int) -> str:
        return doc[pno - 1].get_label() or "–"

    lines.append("### 2.1 Page range of the primary care chapters\n")
    lines.append(
        f"Part divider *PRIMARY HEALTH CARE* on PDF page {toc[i_pc][2]}; chapters run from PDF page "
        f"**{chapters[0][2]} to {pc_last}** (printed pages {label(chapters[0][2])}–{label(pc_last)}). "
        f"PDF page = printed page + {chapters[0][2] - int(label(chapters[0][2]))}.\n"
    )
    rows = []
    for k, (_, title, start) in enumerate(chapters):
        end = (chapters[k + 1][2] - 1) if k + 1 < len(chapters) else pc_last
        rows.append([title, f"{start}–{end}", f"{label(start)}–{label(end)}"])
    lines.append(md_table(["chapter", "PDF pages", "printed pages"], rows))

    # --- table 23.5 with pdfplumber --------------------------------------- #
    uti_start = next(p for _, t, p in chapters if t.startswith("23."))
    t_pages = [p for p in range(uti_start, pc_last + 1) if "Table 23.5" in doc[p - 1].get_text()]
    lines.append("\n### 2.2 Lower UTI dosing table (Table 23.5) with pdfplumber\n")
    lines.append(
        f"Table 23.5 spans PDF pages {', '.join(map(str, t_pages))} (printed "
        f"{', '.join(label(p) for p in t_pages)}). `pdfplumber` `extract_tables()` with default settings "
        "returns:\n"
    )
    row_starts: list[str] = []
    with pdfplumber.open(src.path, pages=t_pages) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            lines.append(f"**PDF page {page.page_number}** – {len(tables)} table(s) detected\n")
            for ti, table in enumerate(tables):
                lines.append(f"Table {ti}, {len(table)} rows:\n")
                width = max(len(r) for r in table)
                lines.append(
                    md_table(
                        [f"col {c}" for c in range(width)],
                        [[(c or "").replace("\n", " ⏎ ") if c is not None else "None" for c in r] for r in table],
                    )
                )
                lines.append("")
                row_starts += [next((c for c in r if c), "") for r in table]
    epub_drugs: list[str] = epub_facts["epub_uti_drugs"]  # type: ignore[assignment]
    norm = lambda s: re.sub(r"\s+", "", s).lower()  # noqa: E731
    # A drug row is "found" only if some extracted row *starts* with the drug name
    # (a substring test would accept "Trimethoprim" inside "Sulfamethoxazole+trimethoprim").
    missing = [d for d in epub_drugs if not any(norm(s).startswith(norm(d)) for s in row_starts)]
    plain = " ".join(doc[p - 1].get_text() for p in t_pages)
    in_plain = [d for d in missing if norm(d) in norm(plain)]
    lines.append(
        f"**Verdict: not clean.** The EPUB table has {len(epub_drugs)} drug rows ({', '.join(epub_drugs)}); "
        f"pdfplumber's table output is **missing {len(missing)} of {len(epub_drugs)}**: {', '.join(missing) or 'none'}. "
        f"The missing rows are present in the plain page text ({len(in_plain)}/{len(missing)} found with "
        "PyMuPDF `get_text()`), so they are lost by table detection: the table has no ruling lines and only "
        "alternate rows are shaded, so pdfplumber only finds the shaded rows. In the rows it does return, "
        "cells are split across phantom columns (`None` cells), footnote superscripts are fused to words "
        "(`durationd`, `acid a`), and the table is split in two at the page break (*Table 23.5 continued*), "
        "with the header row repeated.\n"
    )

    # --- ligature losses -------------------------------------------------- #
    epub_vocab: set[str] = epub_facts["epub_vocab"]  # type: ignore[assignment]
    pua: Counter = Counter()
    broken: Counter = Counter()
    broken_on_infographic = 0
    for pno in range(pc_first, pc_last + 1):
        text = doc[pno - 1].get_text()
        is_infographic = bool(re.search(r"Page \d of \d", text))
        pua.update(f"U+{ord(c):04X}" for c in PUA_RE.findall(text))
        for w in re.findall(r"[A-Za-z]+", text):
            lw = w.lower()
            if lw in epub_vocab or len(lw) < 2:
                continue
            for lig in ("fi", "fl", "ff", "ffi", "ffl"):
                if any((lw[:i] + lig + lw[i:]) in epub_vocab for i in range(len(lw) + 1)):
                    broken[w] += 1
                    broken_on_infographic += is_infographic
                    break
    lines.append("### 2.3 Other PDF extraction issues (primary care pages)\n")
    lines.append(
        f"- Private Use Area glyphs in extracted text: {dict(pua)} (decorative icons in the running header).\n"
        f"- Words that lose a ligature (a word not in the EPUB vocabulary that becomes one when fi/fl/ff/ffi/ffl "
        f"is inserted): {sum(broken.values())} occurrences, e.g. "
        + ", ".join(f"`{w}`×{n}" for w, n in broken.most_common(10))
        + f". {broken_on_infographic} of the {sum(broken.values())} are on infographic pages (pages "
        "containing *Page N of N*). This heuristic undercounts: words such as *rst*/*first* are caught, "
        "but a loss that leaves another valid word is not.\n"
        "- Every page carries a running header (book title, part name, chapter name) and page number that "
        "must be stripped.\n"
        "- The infographic pages (two per population) are vector graphics, so their text *is* extractable from "
        "the PDF, unlike the EPUB where they are JPEGs – but reading order is box-by-box and needs layout "
        "handling.\n"
    )


# --------------------------------------------------------------------------- #
# Hypertension PDF
# --------------------------------------------------------------------------- #
@dataclass
class RawLine:
    page: int
    text: str
    x0: float
    y0: float
    y1: float
    size: float
    color: int
    horizontal: bool


def htn_lines(doc: pymupdf.Document) -> list[RawLine]:
    out = []
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                s = max(l["spans"], key=lambda sp: len(sp["text"].strip()))
                text = "".join(sp["text"] for sp in l["spans"])
                horizontal = abs(l["dir"][0] - 1) < 1e-3
                out.append(
                    RawLine(page.number + 1, text, l["bbox"][0], l["bbox"][1], l["bbox"][3], round(s["size"], 1), s["color"], horizontal)
                )
    return out


def inspect_hypertension(lines: list[str]) -> None:
    src = get_source("hypertension_pdf")
    doc = pymupdf.open(src.path)
    raw = htn_lines(doc)
    lines.append("## 3. Hypertension guideline – PDF\n")
    lines.append(
        f"File: `data/raw/{src.filename}`, {doc.page_count} pages, A4 portrait. Creator *{doc.metadata.get('creator')}*, "
        f"producer *{doc.metadata.get('producer')}*: this is the NCBI Bookshelf copy (NBK573631) *printed from a "
        "web browser*, not WHO's original InDesign export. It has no outline and no page labels. The ISBN page "
        "gives ISBN 978-92-4-003398-6 (electronic) and the citation *Guideline for the pharmacological "
        "treatment of hypertension in adults. Geneva: World Health Organization; 2021. Licence: CC BY-NC-SA 3.0 "
        "IGO*, but **prints no IRIS handle or URL** (only *CIP data are available at http://apps.who.int/iris*).\n"
    )

    # printed page numbers: lone numerals at the page foot
    printed: dict[int, str] = {}
    for r in raw:
        if r.y0 > 800 and re.fullmatch(r"\s*([0-9]+|[ivxlc]+)\s*", r.text):
            printed[r.page] = r.text.strip()
    h1s: list[tuple[int, str]] = []
    for r in raw:
        if r.size > 20 and r.color == HTN_RED:
            if h1s and h1s[-1][0] == r.page:
                h1s[-1] = (r.page, h1s[-1][1] + " " + r.text.strip())
            else:
                h1s.append((r.page, r.text.strip()))
    rec_i = next(i for i, (_, t) in enumerate(h1s) if t.startswith("3 Recommendations"))
    rec_first, rec_last = h1s[rec_i][0], h1s[rec_i + 1][0] - 1
    lines.append("### 3.1 Top-level structure and the recommendations section\n")
    lines.append(
        md_table(
            ["PDF page", "printed page", "chapter heading (24.9 pt, red #B6294C)"],
            [[p, printed.get(p, "–"), t] for p, t in h1s],
        )
    )
    lines.append(
        f"\n**Recommendations section (*3 Recommendations*) = PDF pages {rec_first}–{rec_last}** (printed pages "
        f"{printed.get(rec_first)}–{printed.get(rec_last)}). PDF page = printed page + "
        f"{rec_first - int(printed[rec_first])} in the main body. The recommendations are also restated in the "
        "Executive summary (printed pp. vii–ix) and summarised graphically in *6 Implementation tools* "
        "(Figs 3–6).\n"
    )

    # recommendation boxes
    lines.append("### 3.2 How recommendations are labelled, and how strength and certainty are shown\n")
    rec_heads: list[tuple[int, str]] = []
    grades: list[tuple[int, str]] = []
    exec_grades: list[tuple[int, str]] = []
    exec_page = next(p for p, t in h1s if t.startswith("Executive summary"))
    prev_white = False
    for r in raw:
        if not r.horizontal:
            continue
        in_rec = rec_first <= r.page <= rec_last
        # Box titles are white text; Fig. 2 also has white labels, so require the
        # title to start "N. RECOMMENDATION" (continuation lines follow directly).
        is_title = re.match(r"^\d+\. RECOMMENDATION", r.text.strip()) is not None
        if in_rec and r.color == HTN_WHITE and r.y0 < 800 and (is_title or prev_white):
            if prev_white and rec_heads:
                rec_heads[-1] = (rec_heads[-1][0], rec_heads[-1][1] + " " + r.text.strip())
            else:
                rec_heads.append((r.page, r.text.strip()))
            prev_white = True
            continue
        prev_white = False
        if GRADING_RE.match(r.text.strip()):
            if in_rec:
                grades.append((r.page, r.text.strip()))
            elif exec_page <= r.page < h1s[[p for p, _ in h1s].index(exec_page) + 1][0]:
                exec_grades.append((r.page, r.text.strip()))
    lines.append(
        "Each recommendation sits in a coloured box. The box title is white text on red and is numbered "
        "1–8 in upper case (`N. RECOMMENDATION(S) ON ...`); there are no labels such as *Recommendation 1a*. "
        "Several boxes contain more than one recommendation statement, each followed by its own grading line "
        "(italic in print, but Type3 fonts give no italic flag):\n"
    )
    lines.append(md_table(["PDF page", "box title"], [[p, t] for p, t in rec_heads]))
    lines.append(
        f"\nGrading lines follow the pattern `<Strong|Conditional> recommendation, <certainty> evidence` "
        f"({len(grades)} in section 3, {len(exec_grades)} in the Executive summary). Strength is *Strong* or "
        "*Conditional*; certainty is one of *high*, *moderate*, *low* or a range (*moderate- to high-certainty*). "
        "Statements also follow the GRADE wording convention *WHO recommends* (strong) vs *WHO suggests* "
        "(conditional). Section 2.5 defines the terms.\n"
    )
    rows = []
    for k in range(max(len(grades), len(exec_grades))):
        g = grades[k] if k < len(grades) else (None, "")
        e = exec_grades[k] if k < len(exec_grades) else (None, "")
        rows.append([k + 1, g[0], g[1], e[1], "" if g[1] == e[1] else "**MISMATCH**"])
    lines.append(md_table(["#", "PDF page", "section 3 grading", "Executive summary grading", ""], rows))
    lines.append(
        "\n**Source inconsistency:** the third statement of Recommendation 1 (*WHO suggests ... without "
        "cardiovascular disease but with high cardiovascular risk ...*) is graded *Strong* in section 3.1 but "
        "*Conditional* in the Executive summary; the verb *suggests* points to conditional. The extraction is "
        "correct – the grading printed beneath the statement really is *Strong*. The guideline group's journal "
        "summary (Al-Makki et al., *Hypertension* 2022;79:293–301, Figure 3; `data/raw/hyp-79-293.pdf`) also "
        "prints *Conditional*. Resolved as conditional: chunks keep `strength_as_printed: \"strong\"`, store "
        "`strength: \"conditional\"`, and record the evidence in `source_discrepancy`.\n"
    )

    # columns
    lines.append("### 3.3 Layout and extraction problems\n")
    body = [r for r in raw if r.horizontal and r.size == 10.0 and r.color == HTN_BODY and r.text.strip()]
    x_hist = Counter(round(r.x0) for r in body)
    right_half = sum(1 for r in body if r.x0 > doc[0].rect.width / 2)
    lines.append(
        f"- **Single column.** Of {len(body)} body-text lines, the most common left edges are "
        + ", ".join(f"x={x} ({n})" for x, n in x_hist.most_common(4))
        + f" (odd/even page margins and list indents); only {right_half} start in the right half of the page, "
        "all of them inside figures, tables or recommendation boxes.\n"
    )
    grey = Counter(r.text.strip() for r in raw if r.color == HTN_GREY)
    side = Counter(r.text.strip() for r in raw if not r.horizontal)
    lines.append(
        f"- **Running header/footer:** {sum(grey.values())} lines in grey #6B6E71: "
        + "; ".join(f"*{t}* ×{n}" for t, n in grey.most_common(3))
        + f". **Page numbers:** {len(printed)} pages carry a numeral in a coloured tab at the foot (white text). "
        f"**Side tabs:** {sum(side.values())} rotated section labels in the outer margin ("
        + ", ".join(f"*{t}*" for t in list(side)[:6])
        + ", ...). All are removable by colour, position or text direction.\n"
    )
    pua_hashes = type3_pua_glyph_hashes(doc)
    by_cp: dict[str, set[str]] = defaultdict(set)
    fonts_by_cp: dict[str, set[int]] = defaultdict(set)
    for (fx, ch), h in pua_hashes.items():
        by_cp[f"U+{ord(ch):04X}"].add(h)
        fonts_by_cp[f"U+{ord(ch):04X}"].add(fx)
    n_type3 = len({fx for fx, _ in pua_hashes} | set())
    examples: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for page in doc:
        for b in page.get_text("rawdict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = "".join(c["c"] for c in s["chars"])
                    m = re.search(r"\((\d+) 0 R\)", s["font"])
                    for hit in re.finditer(r"\w*[\ue000-\uf8ff]\w*", t):
                        ch = PUA_RE.search(hit.group()).group()
                        h = pua_hashes.get((int(m.group(1)) if m else -1, ch))
                        examples[(f"U+{ord(ch):04X}", h)][hit.group().replace(ch, "◆")] += 1
    rows = [[cp, h, sum(c.values()), ", ".join(w for w, _ in c.most_common(5))] for (cp, h), c in sorted(examples.items())]
    lines.append(
        "- **Ligatures and accents are Private Use Area code points whose meaning changes from font to font.** "
        f"The browser print embedded text as Type3 fonts (one or more per page; {len({fx for fx, _ in pua_hashes})} "
        "of them map something to the PUA). `fi`, `fl` and spacing accents are mapped to U+E01E/U+E01F/U+E060/"
        "U+E0A8/U+E0B4/U+E100. Default `get_text()` shows them as nothing, producing *specic*, *conrmed*, "
        "*benets*, *Ofce*. Worse, the same code point means `fi` in one font and `fl` in another, so a fixed "
        "code-point table would corrupt words. Identifying each glyph by the MD5 of its Type3 drawing procedure "
        "separates them cleanly (◆ = the PUA glyph):\n"
    )
    lines.append(md_table(["code point", "glyph hash", "occurrences", "example words"], rows))
    lines.append(
        "\n  Reading the examples: hashes `6a23a277`, `169a766e`, `8ffb1680` are **fi**; `e71e9475`, `f0f57f4b` are "
        "**fl**; `a846d610`, `2dce5ce9` are a spacing acute accent, `3153d706`, `29b7749d` a grave and "
        "`a0fbcf41` a diaeresis, each *following* its base letter; `44d02e14` is a dotless ı used as the base "
        "of *í* (*Martí*, checked visually). Accented glyphs occur only in names (acknowledgements, references, "
        "annexes).\n"
    )
    bullets = sum(1 for r in raw if r.text.strip() == "[")
    inline_bullets = sum(1 for r in raw if r.text.startswith("[ "))
    hyph = [r for r in body if re.search(r"[A-Za-z]-\s*$", r.text)]
    hyph_examples = Counter(re.search(r"(\S+-)\s*$", r.text).group(1) for r in hyph)
    flags = Counter(
        s["flags"] for p in doc for b in p.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
    )
    lines.append(
        f"- **Bullets** are a symbol glyph extracted as `[`: {bullets} stand-alone `[` lines (bullet drawn as a "
        f"separate span beside the text) plus {inline_bullets} lines starting `[ `. They must be converted to a "
        "list marker by position, not by character, because `[` also occurs legitimately (e.g. *[website]*).\n"
        f"- **Line-end hyphens:** {len(hyph)} body lines end in `letter-`. Spot checks show almost all are real "
        "compound hyphens split at the line end ("
        + ", ".join(f"`{w}`" for w, _ in hyph_examples.most_common(8))
        + "), so the hyphen must be kept by default and only removed when the joined word is attested "
        "elsewhere in the document.\n"
        f"- **No font-style information:** span flags are {dict(flags)}; Type3 fonts carry no bold/italic flags, so "
        "headings must be detected by size and colour (chapter 24.9 pt red, section 12 pt red, sub-heading 10 pt "
        "red, recommendation box title white) and grading lines by text pattern.\n"
        "- **Broken lines in headings:** long chapter and section titles wrap (e.g. *2 Method for developing the / "
        "guideline*, *3.3 Cardiovascular disease risk assessment as guide to initiation of / antihypertensive "
        "medications*); consecutive heading-styled lines must be merged.\n"
        "- **Figures are text, not images.** Figs 1–6 and Fig. A3.1 are vector drawings whose labels extract as "
        "loose lines in drawing order (e.g. Fig. 2's flow chart, Fig. 5/6's dose-titration boxes). Figs 5 and 6 "
        "contain the only drug- and dose-specific content in the guideline (telmisartan, amlodipine, "
        "hydrochlorothiazide, chlorthalidone doses), so they need layout-aware reading order rather than being "
        "dropped. Table 1 and the Annex 4 PICO tables are two-column layouts without ruling lines.\n"
        f"- Pages with no text at all: {', '.join(str(p.number + 1) for p in doc if not p.get_text().strip())}.\n"
    )


# --------------------------------------------------------------------------- #
# Recommendation and report assembly
# --------------------------------------------------------------------------- #
RECOMMENDATION = """## 4. Recommendation: primary source for AWaRe

**Use the EPUB as the primary source for the AWaRe primary care chapters, and the PDF only as a
secondary source for page verification and (optionally) the infographic pages.**

Reasons, all measured above:

1. **Dosing tables are real tables in the EPUB.** Table 23.5 is a `<table>` with one `<tr>` per drug
   and one `<td>` per population, so adult/child/duration cells are unambiguous. pdfplumber silently
   drops half of the drug rows of the same table from the PDF (the unshaded ones), fuses footnote
   markers into words and splits the table at the page break. A dosing-accuracy evaluation cannot be
   built on silently missing rows.
2. **Structure is explicit.** One XHTML file per chapter, `h1`/`h2`/`h3` for chapter/section/subsection,
   `<caption>` for table titles and `<tfoot>` for table notes. In the PDF all of this would have to be
   inferred from font size, colour and position.
3. **AWaRe category survives.** Access/Watch/Reserve is a CSS class (`back_verde`, ...) in the EPUB; in
   the PDF it is only a background colour drawn behind the drug name.
4. **Clean Unicode text.** No ligature losses, PUA icons or running headers to strip.
5. **Citations are not lost.** `epub:type="pagebreak"` markers carry the printed page numbers, which
   match the PDF page labels, so chunks can still cite printed pages.

Caveats to handle:

- The one-page clinical **infographics are JPEG-only in the EPUB**. They mostly restate the chapter
  text and tables, but contain a few short summary statements not phrased identically elsewhere. Either
  skip them (simplest, no loss of dosing data, since every regimen is in the HTML tables) or extract
  those pages from the PDF with layout handling. Recommended: skip them initially and record this as a
  known gap.
- The EPUB (ISBN 978-92-4-006937-4) and PDF (ISBN 978-92-4-006238-2) are different editions of the same
  2022 book; their Table 23.5 content matches, but a spot check of each chapter's dosing table against the
  PDF is worth adding to the AWaRe parser's tests.
- Markup is hand-edited (e.g. `<colgroup>` with 2 columns for a 3-column table), so the parser should
  read cells from `<tr>/<td>`, not from `<colgroup>`.
"""


def main() -> None:
    lines: list[str] = [
        "# Source inspection report",
        "",
        "_Generated by `scripts/inspect_sources.py`. Do not edit by hand; re-run the script instead._",
        "",
        "Sources inspected: the WHO AWaRe antibiotic book (EPUB and PDF, primary care chapters only) and the "
        "WHO hypertension guideline (PDF). Page numbers are 1-based PDF page indices unless stated as *printed*.",
        "",
    ]
    epub_facts = inspect_epub(lines)
    vocab: set[str] = set()
    for d in epub_facts["docs"]:  # type: ignore[union-attr]
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]+", html.unescape(d.soup.get_text(" "))))
    epub_facts["epub_vocab"] = vocab
    inspect_aware_pdf(lines, epub_facts)
    inspect_hypertension(lines)
    lines.append(RECOMMENDATION)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
