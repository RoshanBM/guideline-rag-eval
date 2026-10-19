# guideline-rag-eval

Retrieval-augmented QA over clinical guidelines, with evaluation of retrieval quality, faithfulness and refusal.

## Layout

```
data/raw/            # PDFs/HTML (git-ignored)
data/chunks/         # chunks.jsonl per collection
eval/questions.jsonl
src/
  parse/             # who_bookshelf.py, aware_pdf.py
  retrieve/          # bm25.py, dense.py, hybrid.py, rerank.py
  generate/          # answer.py (quote + cite + refuse)
  tools/             # dose_lookup.py (structured table query)
  evaluate/          # retrieval_metrics.py, faithfulness.py, refusal.py
app/streamlit_app.py
```

## Results

_TBD_

## Failure cases

_TBD_

## Licences

_TBD_
