"""Run one or more retrievers over every eval item and score them.

Usage::

    python scripts/run_retrieval_eval.py --retriever bm25 [bm25-nostem bge-small ... | all]
                                         [--no-stem] [--questions eval/questions.jsonl]
                                         [--chunks-dir data/chunks] [--out-dir results/retrieval]

Retrievers are the names in ``src/retrieve/registry.py``: ``bm25`` (stemmed baseline),
``bm25-nostem`` (ablation; ``--retriever bm25 --no-stem`` is the same), ``bge-small``,
``bge-base``, ``pubmedbert``, ``medcpt``; ``all`` runs every one.

For each **answerable** item it records, per gold source, the chunks holding its span and the
best rank among them, the rank of each relevant chunk (null if not in the top ``DEPTH`` = 50),
the top-10 chunk ids with scores and the per-item metrics of
``src/evaluate/retrieval_metrics.py`` (recall is per gold source). For each **unanswerable**
item it records the top-10 chunks and the top score (input for a later refusal threshold);
these items are excluded from the metrics.

Per retriever it writes ``<out-dir>/<retriever>.json`` (per-item results, summaries with 95 %
bootstrap CIs, timing, device, and for dense models the truncation report) and appends one row
to ``<out-dir>/summary.csv``. ``scripts/compare_retrievers.py`` compares the JSON files.

Timing: the index is built from scratch (BM25: build time; dense: corpus encode time, model load
and a warm-up batch excluded). One warm-up query is run before timing; per-query latency is the
wall time of ``search(question, DEPTH)`` (dense: query encoding plus FAISS search).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.bootstrap import N_RESAMPLES, SEED, bootstrap_ci  # noqa: E402
from src.evaluate.eval_schema import EvalItem, parse_item  # noqa: E402
from src.evaluate.relevance import load_chunks, relevant_chunk_ids_by_source  # noqa: E402
from src.evaluate.retrieval_metrics import METRIC_NAMES, item_metrics, mean_metrics  # noqa: E402
from src.retrieve.base import Retriever  # noqa: E402
from src.retrieve.corpus import CorpusChunk, load_corpus  # noqa: E402
from src.retrieve.registry import RETRIEVER_NAMES, build_retriever  # noqa: E402

DEPTH = 50
TOP_N = 10
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
CHUNKS_DIR = ROOT / "data" / "chunks"
OUT_DIR = ROOT / "results" / "retrieval"
METRICS_DEFINITION = "v2: recall@k per gold source; hit@k, mrr@10, ndcg@k over relevant chunks"
CI_METRICS: tuple[str, ...] = ("hit@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")
CSV_CI_METRICS: tuple[str, ...] = ("recall@5", "mrr@10", "ndcg@10")
SUMMARY_FIELDS: tuple[str, ...] = (
    "retriever", "date", "git_commit", "questions_sha256", "n_items", "n_answerable", "n_unanswerable",
    *METRIC_NAMES,
    *(f"{m}_ci_{end}" for m in CSV_CI_METRICS for end in ("lo", "hi")),
    "latency_median_ms", "latency_p95_ms", "index_build_s", "device",
)


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def load_items(path: Path) -> list[EvalItem]:
    with path.open(encoding="utf-8") as fh:
        return [parse_item(json.loads(line)) for line in fh if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel_path(path: Path) -> str:
    path = path.resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def run(retriever: Retriever, items: list[EvalItem], all_chunks: list[dict[str, Any]],
        corpus: list[CorpusChunk]) -> tuple[list[dict[str, Any]], list[float]]:
    indexed = {c.chunk_id for c in corpus}
    retriever.search(items[0].question, DEPTH)  # warm-up, not timed
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in items:
        t0 = time.perf_counter()
        hits = retriever.search(item.question, DEPTH)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [cid for cid, _ in hits]
        rank_of = {cid: r for r, cid in enumerate(ranked, start=1)}
        row: dict[str, Any] = {
            "qid": item.qid, "question": item.question, "category": item.category,
            "collection": item.collection, "answerable": item.answerable,
            "latency_ms": round(latencies[-1], 3),
        }
        if item.answerable:
            by_source = relevant_chunk_ids_by_source(item, all_chunks)
            if any(not ids for ids in by_source):
                raise ValueError(f"{item.qid}: a gold source matches no chunk (run validate_eval.py)")
            row["gold_sources"] = [
                {"doc_id": src.doc_id, "printed_page": src.printed_page, "chunk_ids": ids,
                 "best_rank": min((rank_of[c] for c in ids if c in rank_of), default=None)}
                for src, ids in zip(item.gold_sources, by_source)
            ]
            gold = list(dict.fromkeys(cid for ids in by_source for cid in ids))
            row["gold"] = [{"chunk_id": cid, "rank": rank_of.get(cid), "indexed": cid in indexed}
                           for cid in gold]
            row["metrics"] = item_metrics(ranked, by_source)
        row["top_score"] = round(hits[0][1], 4) if hits else None
        row["top"] = [{"chunk_id": cid, "score": round(score, 4)} for cid, score in hits[:TOP_N]]
        results.append(row)
    return results, latencies


def group_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows), **mean_metrics(rows)}
    out["ci95"] = {}
    for m in CI_METRICS:
        ci = bootstrap_ci([r[m] for r in rows])
        out["ci95"][m] = [round(ci.lo, 4), round(ci.hi, 4)]
    return out


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in results if r["answerable"]]
    by_cat: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_coll: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in scored:
        by_cat[r["category"]].append(r["metrics"])
        by_coll[r["collection"]].append(r["metrics"])
    return {
        "overall": group_summary([r["metrics"] for r in scored]),
        "by_category": {k: group_summary(v) for k, v in sorted(by_cat.items())},
        "by_collection": {k: group_summary(v) for k, v in sorted(by_coll.items())},
        "bootstrap": {"n_resamples": N_RESAMPLES, "seed": SEED, "method": "percentile, items resampled"},
    }


def worst_rank(row: dict[str, Any]) -> float:
    return max(s["best_rank"] if s["best_rank"] is not None else DEPTH + 1 for s in row["gold_sources"])


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
SHORT = {name: name.replace("recall", "rec") for name in METRIC_NAMES}


def print_table(title: str, groups: dict[str, dict[str, Any]]) -> None:
    label_w = max(len(title), *(len(k) for k in groups))
    header = f"{title:<{label_w}} {'n':>3} " + " ".join(f"{SHORT[m]:>7}" for m in METRIC_NAMES)
    print(header)
    print("-" * len(header))
    for label, g in groups.items():
        print(f"{label:<{label_w}} {g['n']:>3} " + " ".join(f"{g[m]:>7.3f}" for m in METRIC_NAMES))
    print()


def print_report(out: dict[str, Any], full: bool, n_worst: int = 5) -> None:
    s = out["summary"]
    print(f"\n=== {out['retriever']}  commit={out['git_commit']}  device={out['device']}  "
          f"items={out['n_items']} (answerable {out['n_answerable']}, unanswerable {out['n_unanswerable']})")
    print(f"build/encode {out['index_build_s']:.3f} s; query latency median {out['latency_median_ms']:.2f} ms, "
          f"p95 {out['latency_p95_ms']:.2f} ms")
    if "truncation" in out:
        t = out["truncation"]
        print(f"truncation: {t['n_chunks_truncated']}/{t['n_chunks']} chunks > {t['doc_max_length']} tokens "
              f"(max {t['max_chunk_tokens']}); {t['n_queries_truncated']}/{t['n_queries']} queries > "
              f"{t['query_max_length']}")
    ci = s["overall"]["ci95"]
    print("overall: " + "  ".join(f"{m} {s['overall'][m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]" for m in CI_METRICS))
    if not full:
        return
    print()
    print_table("category", s["by_category"])
    print_table("collection", s["by_collection"])

    answerable = [r for r in out["items"] if r["answerable"]]
    worst = sorted(answerable, key=lambda r: (-worst_rank(r), r["qid"]))[:n_worst]
    print("Worst-ranked answerable items (by the best rank of their worst-found gold source):")
    for r in worst:
        print(f"\n  {r['qid']} [{r['category']}] {r['question']}")
        for g in r["gold_sources"]:
            rank = g["best_rank"] if g["best_rank"] is not None else f"not in top {DEPTH}"
            print(f"    gold {', '.join(g['chunk_ids'])}  rank {rank}")
        for i, t in enumerate(r["top"][:3], start=1):
            print(f"    #{i} {t['chunk_id']}  {t['score']:.3f}")

    top_scores = [r["top_score"] for r in answerable if r["top_score"] is not None]
    print(f"\nTop score: answerable items median {statistics.median(top_scores):.3f} "
          f"(min {min(top_scores):.3f}, max {max(top_scores):.3f}, n={len(top_scores)})")
    for r in out["items"]:
        if not r["answerable"]:
            top1 = r["top"][0]["chunk_id"] if r["top"] else "-"
            score = f"{r['top_score']:.3f}" if r["top_score"] is not None else "none"
            print(f"  {r['qid']:<8} [{r['category']}] top {score}  ({top1})")


# --------------------------------------------------------------------------- #
def append_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not (path.exists() and path.stat().st_size > 0)
    if not new:
        with path.open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != list(SUMMARY_FIELDS):
            raise SystemExit(f"{path} has different columns than this script writes; move it aside")
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def evaluate(name: str, items: list[EvalItem], all_chunks: list[dict[str, Any]], corpus: list[CorpusChunk],
             questions_path: Path, out_dir: Path, commit: str) -> dict[str, Any]:
    built = build_retriever(name, corpus)
    retriever = built.retriever
    truncation = None
    if hasattr(retriever, "truncation_report"):
        truncation = retriever.truncation_report(corpus, [it.question for it in items])
    results, latencies = run(retriever, items, all_chunks, corpus)
    if hasattr(retriever, "close"):
        retriever.close()

    n_ans = sum(r["answerable"] for r in results)
    out: dict[str, Any] = {
        "retriever": name,
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "questions": rel_path(questions_path),
        "questions_sha256": file_sha256(questions_path),
        "n_items": len(results),
        "n_answerable": n_ans,
        "n_unanswerable": len(results) - n_ans,
        "n_indexed_chunks": len(corpus),
        "depth": DEPTH,
        "metrics_definition": METRICS_DEFINITION,
        "device": built.device,
        "config": built.config,
        "index_build_s": round(built.build_s, 4),
        "latency_median_ms": round(float(np.median(latencies)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
    }
    if truncation is not None:
        out["truncation"] = truncation
    out["summary"] = summarise(results)
    out["items"] = results

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    overall = out["summary"]["overall"]
    append_summary(out_dir / "summary.csv", {
        **{k: out[k] for k in ("retriever", "date", "git_commit", "n_items", "n_answerable", "n_unanswerable",
                               "latency_median_ms", "latency_p95_ms", "index_build_s", "device")},
        "questions_sha256": out["questions_sha256"][:12],
        **{m: round(overall[m], 4) for m in METRIC_NAMES},
        **{f"{m}_ci_{end}": overall["ci95"][m][i] for m in CSV_CI_METRICS for i, end in enumerate(("lo", "hi"))},
    })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retriever", required=True, nargs="+", choices=[*RETRIEVER_NAMES, "all"])
    ap.add_argument("--no-stem", action="store_true", help="use the unstemmed BM25 (bm25 -> bm25-nostem)")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # questions contain ≥, – etc.

    names = list(RETRIEVER_NAMES) if "all" in args.retriever else args.retriever
    if args.no_stem:
        names = ["bm25-nostem" if n == "bm25" else n for n in names]
    names = list(dict.fromkeys(names))

    items = load_items(args.questions)
    all_chunks = load_chunks(args.chunks_dir)
    corpus = load_corpus(args.chunks_dir)
    commit = git_commit()
    for name in names:
        out = evaluate(name, items, all_chunks, corpus, args.questions, args.out_dir, commit)
        print_report(out, full=len(names) == 1)
    print(f"\nwrote {', '.join(f'{n}.json' for n in names)} to {rel_path(args.out_dir)}/ and appended to summary.csv")
    if len(names) > 1:
        print("compare with: python scripts/compare_retrievers.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
