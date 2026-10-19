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
| AWaRe parser | pending review of the source inspection |
| Retrieval, generation, evaluation, app | not started |

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
.venv/Scripts/python -m pytest
```

The raw files are expected in `data/raw/` under the file names WHO and NCBI distribute them with
(see the table below). Their file names are mapped to logical source IDs in `src/parse/sources.py`.
[data/raw/MANIFEST.md](data/raw/MANIFEST.md) records each file's source, licence and SHA-256 hash.

## Chunk schema

Each line of a `data/chunks/*.jsonl` file is one JSON object with exactly these keys, in this order:

| Key | Type | Meaning |
|---|---|---|
| `chunk_id` | string | Stable, deterministic ID, e.g. `who-htn-2021-rec-01a`, `who-htn-2021-s3-1-evidence-and-rationale`. |
| `collection` | string | Corpus grouping, e.g. `cardiometabolic`. |
| `doc` | string | Human-readable document title with year. |
| `doc_id` | string | Document ID, e.g. `who-htn-2021`. |
| `section_path` | list of strings | Headings from chapter down, as printed (e.g. `["3 Recommendations", "3.1 Blood pressure threshold …"]`). |
| `rec_id` | string or null | Recommendation label (`"Recommendation 1"`) on recommendation chunks; null otherwise. |
| `rec_part` | int or null | 1-based position of the statement within its recommendation box (a box can hold several statements, each with its own grading); null for non-recommendation chunks. |
| `parent_rec_id` | string or null | On implementation-remark chunks, the `rec_id` the remarks belong to; null otherwise. |
| `chunk_type` | string | `recommendation`, `prose`, `table` or `pico` (PICO question tables from the guideline's methods annex). |
| `index_default` | bool | `false` for `pico` chunks, `true` otherwise; retrieval indexes only `true` chunks unless asked otherwise. |
| `strength` | string or null | Resolved strength, `strong` or `conditional`. Equals `strength_as_printed` unless `source_discrepancy` records a reviewed resolution; null for non-recommendation chunks, and for any unresolved conflict. |
| `strength_as_printed` | string or null | Strength exactly as printed in the grading line beneath the statement; null for non-recommendation chunks. |
| `certainty` | string or null | `high`, `moderate`, `low`, `very low`, or a range such as `moderate to high`, as printed; null for non-recommendation chunks. |
| `source_discrepancy` | string or null | When the source contradicts itself about this chunk: what each part says, the evidence checked, and how it was resolved. Null everywhere else. |
| `text` | string | Clean chunk text (see below). |
| `pdf_page_start`, `pdf_page_end` | int | 1-based page numbers in the source PDF file. |
| `printed_page_start`, `printed_page_end` | int or null | Page numbers printed on those pages, read from each page itself; null if the page prints none. |
| `source_url` | string | Canonical source record. |
| `licence` | string | `CC BY-NC-SA 3.0 IGO`. |

Chunk text is normalised by `src/parse/text_utils.normalise_text` (NFC, ligatures expanded, whitespace
collapsed, running headers/footers removed). Line breaks separate paragraphs and list items; bullets are
`• `, table cells are separated by ` | `. Quotes are verified with `text_utils.quote_in_text`, which
normalises both sides and also folds curly quotes and dash variants.

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
