# Raw source manifest

The files in this folder are not redistributed (git-ignored); this manifest is committed so
that anyone can obtain the same files and check them byte for byte:

```bash
sha256sum data/raw/*.pdf data/raw/*.epub
```

The four WHO works are © World Health Organization, licensed under
[CC BY-NC-SA 3.0 IGO](https://creativecommons.org/licenses/by-nc-sa/3.0/igo) (licence statement
printed on each work's copyright page). The journal article (`hyp-79-293.pdf`) is © 2021 The Authors,
open access under a Creative Commons Attribution licence; it is used only as a reference to check
recommendation gradings and is not chunked.

**Provenance status.** The download location was not recorded at download time. Entries marked
*inferred* are reconstructed from the file name, the file's own metadata and a web search on
2026-10-19; they should be confirmed (and the download URL replaced with the exact one) by whoever
downloaded the files. IRIS (`iris.who.int`) returned HTTP 403 to automated requests, so no IRIS
page could be opened to confirm a handle directly.

| File | Work | SHA-256 | Size (bytes) |
|---|---|---|---|
| `9789240062382-eng.pdf` | The WHO AWaRe (Access, Watch, Reserve) antibiotic book (2022), PDF, ISBN 978-92-4-006238-2 (electronic version) | `4960ecc4fa5bab8281feda656a0da3176dafa1fd21971d87e3b2092d7dac562f` | 10 323 301 |
| `9789240069374-eng.epub` | The WHO AWaRe (Access, Watch, Reserve) antibiotic book (2022), ePub, ISBN 978-92-4-006937-4 (ePub version) | `a8fe9be6e6501de39be459fb672ff775f651125955e31c929c06f28d43952faf` | 81 748 862 |
| `Bookshelf_NBK573631.pdf` | Guideline for the pharmacological treatment of hypertension in adults (2021), ISBN 978-92-4-003398-6 (electronic version) | `1b773f8fb46f4d0e545cee79e21e830633fa131ee36894c4cb18481e830f1ee6` | 3 202 038 |
| `WHO-UCN-NCD-20.1-eng.pdf` | HEARTS D: diagnosis and management of type 2 diabetes, WHO/UCN/NCD/20.1 (published 22 April 2020; no ISBN) | `71b216b19a227e73e0e624274974571a3530e30a0db37a3d10b6d179eefac008` | 1 248 098 |
| `hyp-79-293.pdf` | Al-Makki A, et al. Hypertension pharmacological treatment in adults: a World Health Organization guideline executive summary. *Hypertension* 2022;79:293–301. doi:10.1161/HYPERTENSIONAHA.121.18192 | `942ede105d82e2bfc9251022697c0625d8e92d50b07a47aa7e41841be78097c9` | 1 930 584 |

## `9789240062382-eng.pdf`

- **Source URL (canonical record):** https://iris.who.int/handle/10665/365237 (*inferred*: found by
  web search for the ISBN; the IRIS page itself could not be opened). Publisher page:
  https://www.who.int/publications/i/item/9789240062382.
- **Downloaded from:** not recorded. The file name `9789240062382-eng.pdf` follows WHO's
  ISBN-based download naming (WHO IRIS / who.int publications), so *inferred* WHO IRIS or who.int.
- **Licence:** CC BY-NC-SA 3.0 IGO (copyright page, PDF p. 4).
- **Local file timestamp:** 2026-10-19 15:30 +05:30. PDF metadata: created 2023-11-30, modified 2024-04-22.

## `9789240069374-eng.epub`

- **Source URL (canonical record):** https://iris.who.int/handle/10665/365237 (*inferred*: the
  same IRIS record lists the ePub ISBN alongside the PDF; not opened directly).
- **Downloaded from:** not recorded; file name follows the same WHO naming pattern (*inferred*
  WHO IRIS or who.int).
- **Licence:** CC BY-NC-SA 3.0 IGO (`disclaimer.xhtml`, which also lists the three ISBNs).
- **Local file timestamp:** 2026-10-19 15:27 +05:30. OPF `dcterms:modified`: 2024-04-25.

## `Bookshelf_NBK573631.pdf`

- **Source URL (canonical record):** https://iris.who.int/handle/10665/344424 (handle given in the
  project brief). The PDF itself prints no handle or URL. Indirect corroboration: NCBI Bookshelf
  files this book under the ID `who344424`, i.e. WHO IRIS item 344424.
- **Downloaded from:** NCBI Bookshelf, book section NBK573631
  (https://www.ncbi.nlm.nih.gov/books/NBK573631/), *inferred* from the file name. The PDF metadata
  (creator *Firefox*, producer *Skia/PDF m149*, created 2026-10-19 10:01 UTC) shows it was printed
  to PDF from a web browser rather than downloaded as WHO's original file; this is why its text
  uses browser-generated Type3 fonts (see `notes/source_inspection.md`).
- **Licence:** CC BY-NC-SA 3.0 IGO (copyright page, PDF p. 4).
- **Local file timestamp:** 2026-10-19 15:31 +05:30.

## `hyp-79-293.pdf`

- **Source URL (canonical record):** https://doi.org/10.1161/HYPERTENSIONAHA.121.18192 (DOI printed
  on every page of the article).
- **Downloaded from:** PubMed Central, PMC8654104 (as stated by the project owner, who added the file
  on 2026-10-19). PDF metadata (creator *Firefox*, producer *Skia/PDF m149*, created 2026-10-19
  11:43 UTC) shows it was printed to PDF from a browser.
- **Licence:** "open access article under the terms of the Creative Commons Attribution License"
  (article p. 293; the licence version is not stated in the extracted text).
- **Use in this project:** reference only. Figure 3 (article p. 298, PDF p. 6; an embedded image,
  so not in the text layer) lists every recommendation with its grading and was used to resolve
  the Recommendation 1 grading discrepancy (see `source_discrepancy` in the chunk files).
- **Local file timestamp:** 2026-10-19 17:13 +05:30.

## `WHO-UCN-NCD-20.1-eng.pdf`

- **Source URL (canonical record):** https://www.who.int/publications/i/item/WHO-UCN-NCD-20.1 (verified by
  the project owner). Published 22 April 2020.
- **Downloaded from:** https://iris.who.int/server/api/core/bitstreams/2a0b4f68-7155-4ad1-b543-945791e31830/content
  (WHO IRIS bitstream, as given by the project owner).
- **File:** WHO's original. PDF metadata: creator *Adobe InDesign CS6 (Macintosh)*, producer *Adobe PDF
  Library 10.0.1*, created 2020-04-21 10:12 +01:00, modified 2020-04-21 13:14 +02:00; 35 pages.
- **Licence:** CC BY-NC-SA 3.0 IGO (copyright page, PDF p. 4: "WHO/UCN/NCD/20.1 © World Health
  Organization 2020 Some rights reserved. This work is available under the Creative Commons
  Attribution-NonCommercial-ShareAlike 3.0 IGO licence (CC BY-NC-SA 3.0 IGO ...)").
- **Year:** 2020 (copyright page, suggested citation and publication date); chunk `doc_id` `who-hearts-d-2020`.
- **Local file timestamp:** 2026-10-23 18:44 +05:30.
- **Replaced file:** until 2026-10-23 this name held a Firefox re-print of the same document (creator
  *Firefox*, producer *Skia/PDF m149*; SHA-256
  `67ac38bd24d77d6cb80b40341e0b954891896cc95b69cd309449635fea639633`, 2 244 132 bytes). Parsing the original
  gives identical chunks and page numbers.
