# guideline-rag-eval

A citation-grounded retrieval-augmented generation (RAG) system over WHO clinical guidelines, with
rigorous evaluation of retrieval quality, faithfulness, dosing accuracy and refusal behaviour.

The aim is to answer clinical-guideline questions **only** from retrieved guideline text, to quote
and cite the exact passage (document, section and page) behind every claim, and to refuse when the
guidelines do not support an answer. Each of those properties is measured, not assumed.

> **Not medical advice.** This project is for research and demonstration purposes only. It is not a
> medical device, has not been clinically validated, and must not be used to make decisions about
> the diagnosis or treatment of any person. Always consult the original WHO publications and a
> qualified health professional.

## Status

| Stage | State |
|---|---|
| Project scaffolding | done |
| Source inspection ([notes/source_inspection.md](notes/source_inspection.md)) | done |
| Hypertension guideline parser (`src/parse/who_hypertension.py`) | done |
| AWaRe parser, primary health care chapters (`src/parse/who_aware.py`) | done; dosing sample manually cross-checked against the PDF (see [Manual verification](#manual-verification)) |
| Retrieval, generation, evaluation, app | not started |

## Manual verification

Automated tests cannot prove that values were read correctly from the source, so parts of the
output are checked by hand against the PDFs.

| What | Checked by | Result |
|---|---|---|
| AWaRe dosing sample: 15 rows of `aware_dosing.csv` drawn with `random.Random(42)`, each compared with the printed table on the PDF page given ([notes/aware_verification_sample.md](notes/aware_verification_sample.md)) | project owner, 2026-10-21 | cross-checked against the PDF; no errors reported |
| Same 15 rows with the later columns (`dose_basis`, `resolved_*`, `population_raw`, `dose_unit_note`) ([notes/aware_verification_sample_v2.md](notes/aware_verification_sample_v2.md)) | – | not yet checked |

The tick boxes (`verified: [ ]`) in the sample files have not been filled in; ticking them records
which rows were checked. The parser never rewrites a sample file: new samples are written only with
`--sample <new file>`.

Also checked against rendered PDF pages during development (not a substitute for the owner's check):

- Hypertension Recommendation 1, statement 3: printed *Strong* in section 3.1 (p. 7) and *Conditional* in
  the Executive summary (p. viii) and in Al-Makki et al., *Hypertension* 2022, Figure 3; resolved as
  conditional (see `source_discrepancy`).
- Hypertension Figs 5 and 6 (dose-titration algorithms): step order and doses read off the page renders.
- Hypertension ligature/accent glyphs: each glyph shape checked against the words it occurs in.
- AWaRe: the "3–6 kg" amoxicillin band (Tables 5.3, 6.5) and the Table 18.2 caption confirmed in the PDF;
  every dose and antibiotic string in `aware_dosing.csv` is also found automatically on its PDF page.

## Layout

```
data/raw/            # source PDFs/EPUB (git-ignored, not redistributed)
data/interim/        # intermediate extraction artefacts
data/chunks/         # one JSONL file of chunks per source document
eval/                # evaluation question sets
notebooks/           # exploration only, never core logic
notes/               # design notes and source inspection report
scripts/             # one-off utilities (e.g. inspect_sources.py)
src/
  parse/             # source-specific parsers + shared text normalisation (text_utils.py)
  retrieve/          # retrieval (BM25 / dense / hybrid / rerank)
  generate/          # answer generation (quote + cite + refuse)
  tools/             # structured tools, e.g. dose lookup
  evaluate/          # retrieval, faithfulness, dosing and refusal metrics
app/                 # demo app
tests/               # pytest suite
```

## Quick start

```bash
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python scripts/inspect_sources.py              # regenerates notes/source_inspection.md
.venv/Scripts/python -m src.parse.who_hypertension           # writes data/chunks/hypertension.jsonl
.venv/Scripts/python -m src.parse.who_aware                  # writes data/chunks/aware.jsonl, data/interim/aware_*.csv
#   add --sample notes/<new file>.md to write a new verification sample (never overwrites an existing file)
.venv/Scripts/python -m pytest
```

The raw files are expected in `data/raw/` under the file names WHO and NCBI distribute them with
(see the table below). Their file names are mapped to logical source IDs in `src/parse/sources.py`.
[data/raw/MANIFEST.md](data/raw/MANIFEST.md) records each file's source, licence and SHA-256 hash.

## Chunk schema

All collections share one schema, defined and validated in [src/parse/schema.py](src/parse/schema.py).
Each line of a `data/chunks/*.jsonl` file is one JSON object with exactly these keys, in this order
(fields that do not apply to a collection are null):

| Key | Type | Meaning |
|---|---|---|
| `chunk_id` | string | Stable, deterministic ID, e.g. `who-htn-2021-rec-01a`, `who-htn-2021-s3-1-evidence-and-rationale`. |
| `collection` | string | Corpus grouping, e.g. `cardiometabolic`. |
| `doc` | string | Human-readable document title with year. |
| `doc_id` | string | Document ID, e.g. `who-htn-2021`. |
| `section_path` | list of strings | Headings from chapter down, as printed (e.g. `["3 Recommendations", "3.1 Blood pressure threshold …"]`). AWaRe paths start `["Primary health care", "23. Lower urinary tract infection", …]`. |
| `infection` | string or null | *AWaRe.* The chapter's condition, e.g. `Lower urinary tract infection`. |
| `population` | string or null | *AWaRe.* Normalised population: `adults`, `adults_adolescents`, `children`, `neonates`, `pregnant` or `all` (adults and children), where a table column, in-table heading, row label or caption names one (a label naming pregnancy, e.g. "Syphilis in pregnancy", maps to `pregnant`); otherwise null. |
| `population_raw` | string or null | *AWaRe.* The verbatim label `population` was derived from, e.g. `Adults`, `Neonates and children`, `adults, adolescents`, `ophthalmia neonatorum`, `Infants`. Null exactly when `population` is null. |
| `context_header` | string or null | *AWaRe.* Parser-written one-line header for dosing-table and `no_antibiotic` chunks, so they make sense when retrieved alone, e.g. `Lower urinary tract infection - adults - antibiotic treatment (equal options):`. **Not source text.** |
| `rec_id` | string or null | Recommendation label (`"Recommendation 1"`) on recommendation chunks; null otherwise. |
| `rec_part` | int or null | 1-based position of the statement within its recommendation box (a box can hold several statements, each with its own grading); null for non-recommendation chunks. |
| `parent_rec_id` | string or null | On implementation-remark chunks, the `rec_id` the remarks belong to; null otherwise. |
| `chunk_type` | string | `recommendation`, `prose`, `table` or `pico` (PICO question tables from the guideline's methods annex). AWaRe adds `no_antibiotic`: passages stating that antibiotics are not needed or not recommended (safety content). |
| `index_default` | bool | `false` for `pico` chunks, `true` otherwise; retrieval indexes only `true` chunks unless asked otherwise. |
| `strength` | string or null | Resolved strength, `strong` or `conditional`. Equals `strength_as_printed` unless `source_discrepancy` records a reviewed resolution; null for non-recommendation chunks, and for any unresolved conflict. |
| `strength_as_printed` | string or null | Strength exactly as printed in the grading line beneath the statement; null for non-recommendation chunks. |
| `certainty` | string or null | `high`, `moderate`, `low`, `very low`, or a range such as `moderate to high`, as printed; null for non-recommendation chunks. |
| `source_discrepancy` | string or null | When the source contradicts itself about this chunk: what each part says, the evidence checked, and how it was resolved. Null everywhere else. |
| `source_text` | string | Verbatim source content only (see below). Quotes are verified against this plus the chunk's footnotes. |
| `aware_category_note` | string or null | *AWaRe dosing chunks.* The Access/Watch/Reserve category of each antibiotic in the chunk, e.g. `AWaRe category (derived from the colour coding in the source): Nitrofurantoin – ACCESS`. The book shows the category **only by colour** (green = ACCESS, orange = WATCH, red = RESERVE, per the legend under each table); this note is derived from that colour coding and is **not source text**. |
| `text` | string | Retrieval text: `context_header` (if any), newline, `source_text`, then `Notes:` followed by the verbatim footnotes the chunk uses (if any), then `aware_category_note` (if any). For the hypertension guideline `text` equals `source_text`. |
| `pdf_page_start`, `pdf_page_end` | int | 1-based page numbers in the source PDF file. |
| `printed_page_start`, `printed_page_end` | int or null | Page numbers printed on those pages, read from each page itself; null if the page prints none. |
| `source_url` | string | Canonical source record. |
| `licence` | string | `CC BY-NC-SA 3.0 IGO`. |

The AWaRe book has no graded recommendations, so its `rec_id`, `rec_part`, `parent_rec_id`, `strength`,
`strength_as_printed` and `certainty` are always null. Its dosing tables are also exported row by row to
`data/interim/aware_dosing.csv` (one row per antibiotic × route/sub-group × weight band, values copied
verbatim, with the same `population`/`population_raw` scheme; an in-table sub-group such as
"• Neonates:" narrows a row's population) and its no-antibiotic statements to
`data/interim/aware_no_antibiotic.csv`. Besides the verbatim columns, each dosing row has:

- `dose_basis`: `per_day` (dose printed as mg/kg/day), `course` (weekly schedule), `single` ("single dose"
  in the frequency or duration) or `per_dose` (everything else);
- `resolved_dose`, `resolved_frequency`, `resolved_from`, `resolved_rule`: usable dose and frequency, with
  the rule that produced them: `as_printed`; `adult_dose_lookup` ("use adult dose" rows, linked to the adult
  row in `resolved_from`); `from_duration` (no frequency printed, schedule taken from the duration:
  "single dose" -> `once`, "one dose per week" -> `once weekly`); `per_day_total` (daily total, frequency
  left empty); `not_stated`; `unresolved`.

Text is normalised by `src/parse/text_utils.normalise_text` (NFC, ligatures expanded, whitespace
collapsed, running headers/footers removed). Line breaks separate paragraphs and list items; bullets are
`• `, table cells are separated by ` | `. Footnote marker letters are removed inline; the full text of
each footnote a chunk uses is appended under `Notes:` (the colour legend is replaced by
`aware_category_note`). Quotes must be verified with `schema.quote_verifies(quote, record)`, which
checks only `source_text` and the footnotes, never `context_header` or `aware_category_note`; it uses
`text_utils.quote_in_text`, which normalises both sides and also folds curly quotes and dash variants.

Where the source contradicts itself, the conflict is resolved by review, never silently: the printed
value stays in `strength_as_printed`, the resolved value goes in `strength`, and the reasoning and
evidence go in `source_discrepancy`. Resolutions are listed in `GRADING_RESOLUTIONS` in
`src/parse/who_hypertension.py`; any conflict not listed there yields `strength: null`. Known case:
the third statement of Recommendation 1 is printed as *Strong* beneath the statement (p. 7) but as
*Conditional* in the Executive summary (p. viii) and in the guideline group's journal summary
(Al-Makki et al., *Hypertension* 2022), and uses the conditional wording "WHO suggests". It is
resolved as conditional.

## Data sources and licences

All guideline content is © World Health Organization and is used under the
[Creative Commons Attribution-NonCommercial-ShareAlike 3.0 IGO licence (CC BY-NC-SA 3.0 IGO)](https://creativecommons.org/licenses/by-nc-sa/3.0/igo).
WHO does not endorse this project, and nothing here should be taken to suggest that it does.
Derived chunk files in `data/chunks/` are adaptations of the WHO works and are distributed under
the same licence. They are for non-commercial use only.

| Logical ID | File in `data/raw/` | Citation |
|---|---|---|
| `aware` (EPUB) | `9789240069374-eng.epub` | The WHO AWaRe (Access, Watch, Reserve) antibiotic book. Geneva: World Health Organization; 2022. Licence: CC BY-NC-SA 3.0 IGO. ISBN 978-92-4-006937-4 (ePub version). |
| `aware` (PDF) | `9789240062382-eng.pdf` | The WHO AWaRe (Access, Watch, Reserve) antibiotic book. Geneva: World Health Organization; 2022. Licence: CC BY-NC-SA 3.0 IGO. ISBN 978-92-4-006238-2 (electronic version). |
| `hypertension` | `Bookshelf_NBK573631.pdf` | Guideline for the pharmacological treatment of hypertension in adults. Geneva: World Health Organization; 2021. Licence: CC BY-NC-SA 3.0 IGO. ISBN 978-92-4-003398-6 (electronic version). Copy obtained via NCBI Bookshelf (NBK573631). |

Only the *Primary health care* chapters (4. Bronchitis to 23. Lower urinary tract infection) of the
AWaRe antibiotic book are used.

Changes made to the source material: text was extracted, cleaned (ligatures, hyphenation and
whitespace normalised; running headers, footers and page numbers removed) and split into
chunks. No clinical content has been added or reworded.

## Results

_TBD_

## Failure cases

_TBD_
