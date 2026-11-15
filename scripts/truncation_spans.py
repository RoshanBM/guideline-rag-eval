"""Where do gold spans sit in truncated gold chunks? Token positions vs each model's limit.

Usage::

    python scripts/truncation_spans.py [--questions eval/questions.jsonl] [--out-dir results/retrieval]

For every dense model (``medcpt-raw`` tokenises exactly like ``medcpt`` and is skipped), every
gold chunk that is longer than the model's max sequence length, and every gold span that chunk
holds, it reports the span's first and last token position in the model input, as the model
sees it: special tokens included, position 0 = [CLS]; for MedCPT the input is the pair
``[context_header, rest of text]``, so positions count the header segment too. With a limit of
512 the last kept content token is position 510 (511 is the final [SEP]); MedCPT truncates the
longer segment (the body), so the same holds there.

Status: ``inside`` (whole span within the kept tokens), ``cut`` (starts inside, ends after the
limit), ``beyond`` (starts after the limit: the model never sees it), or ``not in embedded
text``: the span is not in the chunk's retrieval ``text`` at all (flowchart chunks embed the
derived ``flow_text``, not the verbatim ``source_text`` the gold span is taken from).

The span is located in the chunk's retrieval ``text`` by its words, allowing any whitespace and
any dash or quote variant between/inside them (the text has typo corrections applied, so a span
containing a corrected typo is located by its first words instead; such rows say so).

Writes ``truncation_spans.json`` and ``truncation_spans.txt`` to the output directory. No model
weights are needed beyond the tokenizers, but the encoders are loaded on CPU to use exactly the
tokenizer and limit the retrievers use.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.eval_schema import parse_item  # noqa: E402
from src.evaluate.relevance import load_chunks, relevant_chunk_ids_by_source  # noqa: E402
from src.retrieve.corpus import CorpusChunk, load_corpus  # noqa: E402
from src.parse.text_utils import normalise_for_match  # noqa: E402
from src.retrieve.dense import DENSE_SPECS, _MedCPTEncoder, _STEncoder, make_encoder  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "results" / "retrieval"
MODELS = ("bge-small", "bge-base", "pubmedbert", "medcpt")
DASHES, QUOTES = "-–—‐‑−", "'’‘\"“”"


def span_pattern(words: list[str]) -> re.Pattern[str]:
    """Regex for ``words`` separated by any whitespace, any dash / quote variant matching any other."""
    dash, quote = f"[{re.escape(DASHES)}]", f"[{re.escape(QUOTES)}]"
    parts = ["".join(dash if ch in DASHES else quote if ch in QUOTES else re.escape(ch) for ch in w)
             for w in words]
    return re.compile(r"\s+".join(parts))


def locate(span: str, text: str) -> tuple[int, int, str] | None:
    """(start, end, how) of ``span`` in ``text``: full match, else the longest matching prefix."""
    words = span.split()
    m = span_pattern(words).search(text)
    if m:
        return m.start(), m.end(), "full"
    for n in range(len(words) - 1, 2, -1):
        m = span_pattern(words[:n]).search(text)
        if m:
            return m.start(), m.end(), f"first {n} of {len(words)} words"
    return None


def token_span(enc: Any, chunk: CorpusChunk, start: int, end: int) -> tuple[int, int, int]:
    """(first token position, last token position, total tokens) of chars [start, end) of the body."""
    if isinstance(enc, _MedCPTEncoder):
        header, body = chunk.header_and_body()
        offset = len(chunk.text) - len(body)  # body starts here in chunk.text
        start, end = start - offset, end - offset
        if start < 0:
            raise ValueError("span starts in the context header")
        out = enc.d_tok(header, body, return_offsets_mapping=True)
        seq = out.sequence_ids()
        positions = [i for i, (s, e) in enumerate(out["offset_mapping"]) if seq[i] == 1 and e > start and s < end]
    elif isinstance(enc, _STEncoder):
        out = enc.model.tokenizer(chunk.text, return_offsets_mapping=True, add_special_tokens=True)
        special = set(enc.model.tokenizer.all_special_ids)
        positions = [i for i, ((s, e), tid) in enumerate(zip(out["offset_mapping"], out["input_ids"]))
                     if tid not in special and e > start and s < end]
    else:
        raise TypeError(type(enc))
    return positions[0], positions[-1], len(out["input_ids"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    items = [parse_item(json.loads(line)) for line in args.questions.read_text(encoding="utf-8").splitlines() if line]
    all_chunks = load_chunks()
    corpus = {c.chunk_id: c for c in load_corpus()}
    rows: list[dict[str, Any]] = []
    for name in MODELS:
        enc = make_encoder(DENSE_SPECS[name], "cpu")
        limit = enc.doc_max_length
        last_kept = limit - 2  # final [SEP] occupies position limit - 1
        for item in items:
            if not item.answerable:
                continue
            for src, chunk_ids in zip(item.gold_sources, relevant_chunk_ids_by_source(item, all_chunks)):
                for cid in chunk_ids:
                    chunk = corpus[cid]
                    n_tokens = enc.doc_token_counts([chunk])[0]
                    if n_tokens <= limit:
                        continue
                    row: dict[str, Any] = {"model": name, "qid": item.qid, "chunk_id": cid, "chunk_tokens": n_tokens,
                                           "limit": limit, "span": src.span[:80]}
                    if normalise_for_match(src.span) not in normalise_for_match(chunk.text):
                        # flowcharts embed the derived flow_text, not the verbatim source_text
                        row.update(status=f"not in embedded text ({chunk.chunk_type})")
                        rows.append(row)
                        continue
                    found = locate(src.span, chunk.text)
                    if found is None:
                        row.update(status="not located")
                    else:
                        s, e, how = found
                        first, last, _ = token_span(enc, chunk, s, e)
                        status = "inside" if last <= last_kept else ("cut" if first <= last_kept else "beyond")
                        row.update(span_first_token=first, span_last_token=last, located_by=how, status=status)
                    rows.append(row)
        del enc

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("Gold spans in truncated gold chunks (positions include special tokens; limit 512 keeps "
              "content positions 1-510)\n")
        headers = ["model", "qid", "chunk", "chunk tokens", "span tokens", "status", "located by"]
        table = [[r["model"], r["qid"], r["chunk_id"], str(r["chunk_tokens"]),
                  f"{r.get('span_first_token', '?')}-{r.get('span_last_token', '?')}", r["status"],
                  r.get("located_by", "-")] for r in rows]
        widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("-" * (sum(widths) + 2 * (len(widths) - 1)))
        for t in table:
            print("  ".join(c.ljust(w) for c, w in zip(t, widths)))
    text = buf.getvalue()
    print(text, end="")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "truncation_spans.txt").write_text(text, encoding="utf-8")
    (args.out_dir / "truncation_spans.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
