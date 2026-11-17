"""Write the faithfulness labelling sheet (``notes/faithfulness_labels.csv``).

Usage::

    python scripts/make_faithfulness_sheet.py [--n 30] [--seed 42] [--out notes/faithfulness_labels.csv]

Sample: answered items of the LLM route (``route == "llm"``, valid, not refused; so no refusals,
redirects or template dose answers) from the frozen v2.1 runs on the dev set
(``results/generation/mistral-v2.1.jsonl``) and the held-out set
(``results/generation/heldout/mistral-v2.1.jsonl``). The sample is stratified by collection
(``aware`` / ``cardiometabolic``) with proportional allocation (largest remainder) and drawn with
``random.Random(seed)`` from each stratum's items sorted by item id. Unanswerable items have no
collection; they are assigned by the document of the chunks they cite (AWaRe -> ``aware``,
hypertension or HEARTS-D -> ``cardiometabolic``).

Columns: ``item_id``, ``question``, ``answer``, then per source ``source_<i>_label``,
``source_<i>_quote`` and ``source_<i>_text``, then the empty label columns ``faithful``,
``correct`` and ``comment``. Sources are the answer's citations in order; a label used in the
answer text without a citation is added after them with an empty quote. ``source_<i>_text`` is the
cited chunk's ``source_text`` plus its footnotes (what quotes are verified against). The sheet holds
no automatic judgement (no verification status, route or score).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.relevance import load_chunks  # noqa: E402
from src.parse.schema import verifiable_text  # noqa: E402

RUNS = (
    (ROOT / "results" / "generation" / "mistral-v2.1.jsonl", ROOT / "eval" / "questions.jsonl"),
    (ROOT / "results" / "generation" / "heldout" / "mistral-v2.1.jsonl", ROOT / "eval" / "heldout.jsonl"),
)
OUT = ROOT / "notes" / "faithfulness_labels.csv"
LABEL_COLUMNS = ("faithful", "correct", "comment")
DOC_COLLECTION = {"who-aware-2022": "aware", "who-htn-2021": "cardiometabolic", "who-hearts-d-2020": "cardiometabolic"}
LABEL_IN_TEXT = re.compile(r"S\d+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pool(chunks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Answered LLM-route rows of both runs, each with its question item and collection."""
    out = []
    for results, questions in RUNS:
        items = {it["qid"]: it for it in read_jsonl(questions)}
        for row in read_jsonl(results):
            if row.get("route") != "llm" or not row.get("valid") or row.get("refused"):
                continue
            collection = items[row["qid"]].get("collection")
            if not collection:
                shown = {s["label"]: s["chunk_id"] for s in row["sources"]}
                docs = Counter(DOC_COLLECTION[chunks[shown[c["label"]]]["doc_id"]]
                               for c in row["citations"] if c["label"] in shown)
                collection = docs.most_common(1)[0][0]
            out.append({"row": row, "collection": collection})
    return out


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of ``n`` over strata, rounding by largest remainder."""
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (exact[k] - alloc[k], k), reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def sample(entries: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = {}
    for e in sorted(entries, key=lambda e: e["row"]["qid"]):
        strata.setdefault(e["collection"], []).append(e)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = random.Random(seed)
    chosen = []
    for k in sorted(strata):
        chosen += rng.sample(strata[k], alloc[k])
    return sorted(chosen, key=lambda e: e["row"]["qid"])


def sources_of(row: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    shown = {s["label"]: s["chunk_id"] for s in row["sources"]}
    out = [{"label": c["label"], "quote": c["quote"], "text": verifiable_text(chunks[shown[c["label"]]])}
           for c in row["citations"] if c["label"] in shown]
    cited = {c["label"] for c in row["citations"]}
    for label in dict.fromkeys(LABEL_IN_TEXT.findall(row["answer"] or "")):
        if label in shown and label not in cited:
            out.append({"label": label, "quote": "", "text": verifiable_text(chunks[shown[label]])})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    chunks = {c["chunk_id"]: c for c in load_chunks()}
    entries = pool(chunks)
    chosen = sample(entries, args.n, args.seed)
    rows = [(e["row"], sources_of(e["row"], chunks)) for e in chosen]
    width = max(len(s) for _, s in rows)
    header = ["item_id", "question", "answer"]
    for i in range(1, width + 1):
        header += [f"source_{i}_label", f"source_{i}_quote", f"source_{i}_text"]
    header += list(LABEL_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as fh:  # BOM so Excel reads UTF-8
        w = csv.writer(fh)
        w.writerow(header)
        for row, srcs in rows:
            cells = [row["qid"], row["question"], row["answer"]]
            for s in srcs:
                cells += [s["label"], s["quote"], s["text"]]
            cells += [""] * (len(header) - len(LABEL_COLUMNS) - len(cells)) + [""] * len(LABEL_COLUMNS)
            w.writerow(cells)
    by = Counter(e["collection"] for e in chosen)
    print(f"pool {len(entries)} answered LLM-route items "
          f"({dict(Counter(e['collection'] for e in entries))}); sampled {len(chosen)} ({dict(by)}), seed {args.seed}")
    print(f"dev {sum(not e['row']['qid'].startswith('ho-') for e in chosen)}, "
          f"held-out {sum(e['row']['qid'].startswith('ho-') for e in chosen)}; wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
