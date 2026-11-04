"""Compare retrievers from their ``results/retrieval/<name>.json`` files.

Usage::

    python scripts/compare_retrievers.py [--retrievers bm25 bm25-nostem bge-small ...]
                                         [--baseline bm25] [--results-dir results/retrieval]
                                         [--save results/retrieval/comparison.txt]

All files must come from the same questions file (same SHA-256) and hold the same items.
Prints:

1. one table, rows = retrievers: hit@1, recall@5, recall@10, MRR@10, nDCG@10, each with its
   95 % bootstrap CI (``src/evaluate/bootstrap.py``: 1,000 resamples of items, seed 42), median
   query latency, build/encode time and device; then the same table per collection and per
   category (CIs over very few items are wide; ``n`` is shown);
2. paired bootstrap of every other retriever minus the baseline on recall@5 and MRR@10
   (``*`` marks a CI that excludes 0);
3. answerable items that every retriever misses at k = 10 (no relevant chunk in the top 10),
   and items where every retriever finds some but not all gold sources in the top 10;
4. items where a dense retriever beats the baseline by the largest margin, and the reverse,
   measured by the reciprocal rank of the first relevant chunk within the top 50
   (0 when absent).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.bootstrap import bootstrap_ci, paired_bootstrap  # noqa: E402
from src.retrieve.registry import BM25_NAMES, DENSE_NAMES, RETRIEVER_NAMES  # noqa: E402

RESULTS_DIR = ROOT / "results" / "retrieval"
TABLE_METRICS: tuple[tuple[str, str], ...] = (
    ("hit@1", "hit@1"), ("recall@5", "recall@5"), ("recall@10", "recall@10"),
    ("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10"),
)
PAIRED_METRICS: tuple[str, ...] = ("recall@5", "mrr@10")
DEPTH = 50


def load_results(names: list[str], results_dir: Path) -> dict[str, dict[str, Any]]:
    runs = {}
    for name in names:
        path = results_dir / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}; run scripts/run_retrieval_eval.py --retriever {name}")
        runs[name] = json.loads(path.read_text(encoding="utf-8"))
    shas = {r["questions_sha256"] for r in runs.values()}
    qids = {tuple(i["qid"] for i in r["items"]) for r in runs.values()}
    if len(shas) != 1 or len(qids) != 1:
        raise SystemExit("results come from different question files; re-run them on the same set")
    return runs


def answerable(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [i for i in run["items"] if i["answerable"]]


def first_rank(item: dict[str, Any]) -> int | None:
    ranks = [g["rank"] for g in item["gold"] if g["rank"] is not None]
    return min(ranks) if ranks else None


def rr(item: dict[str, Any]) -> float:
    r = first_rank(item)
    return 1.0 / r if r else 0.0


def fmt_table(headers: list[str], rows: list[list[str]], right_from: int = 1) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) if i < right_from else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))

    return "\n".join([line(headers), "-" * len(line(headers)), *(line(r) for r in rows)])


def metric_cells(items: list[dict[str, Any]]) -> list[str]:
    cells = []
    for key, _ in TABLE_METRICS:
        ci = bootstrap_ci([i["metrics"][key] for i in items])
        cells.append(f"{ci.estimate:.3f} [{ci.lo:.2f}, {ci.hi:.2f}]")
    return cells


def print_overall(runs: dict[str, dict[str, Any]]) -> None:
    headers = ["retriever", "n", *(label for _, label in TABLE_METRICS), "lat ms", "build s", "device"]
    rows = []
    for name, run in runs.items():
        items = answerable(run)
        rows.append([name, str(len(items)), *metric_cells(items), f"{run['latency_median_ms']:.2f}",
                     f"{run['index_build_s']:.2f}", run["device"]])
    print(fmt_table(headers, rows))


def print_grouped(runs: dict[str, dict[str, Any]], key: str, title: str) -> None:
    groups = sorted({i[key] for i in answerable(next(iter(runs.values())))})
    headers = [title, "retriever", "n", *(label for _, label in TABLE_METRICS)]
    rows = []
    for g in groups:
        for name, run in runs.items():
            items = [i for i in answerable(run) if i[key] == g]
            rows.append([g, name, str(len(items)), *metric_cells(items)])
    print(fmt_table(headers, rows, right_from=2))


def print_paired(runs: dict[str, dict[str, Any]], baseline: str) -> None:
    base = answerable(runs[baseline])
    headers = ["retriever", *(f"{m} diff [95% CI]" for m in PAIRED_METRICS)]
    rows = []
    for name, run in runs.items():
        if name == baseline:
            continue
        cells = []
        for m in PAIRED_METRICS:
            ci = paired_bootstrap([i["metrics"][m] for i in answerable(run)], [i["metrics"][m] for i in base])
            cells.append(f"{ci.estimate:+.3f} [{ci.lo:+.3f}, {ci.hi:+.3f}]{' *' if ci.excludes_zero() else '  '}")
        rows.append([f"{name} - {baseline}", *cells])
    print(fmt_table(headers, rows))


def print_misses(runs: dict[str, dict[str, Any]]) -> None:
    names = list(runs)
    by_qid = {name: {i["qid"]: i for i in answerable(run)} for name, run in runs.items()}
    qids = list(by_qid[names[0]])
    missed = [q for q in qids if all(by_qid[n][q]["metrics"]["hit@10"] == 0 for n in names)]
    partial = [q for q in qids if q not in missed
               and all(by_qid[n][q]["metrics"]["recall@10"] < 1 for n in names)]

    def show(q: str) -> None:
        item = by_qid[names[0]][q]
        print(f"  {q} [{item['category']}, {item['collection']}] {item['question']}")
        for src in item["gold_sources"]:
            print(f"    gold: {', '.join(src['chunk_ids'])}")
        ranks = []
        for n in names:
            src_ranks = [s["best_rank"] for s in by_qid[n][q]["gold_sources"]]
            ranks.append(f"{n} {'/'.join('-' if r is None else str(r) for r in src_ranks)}")
        print("    best rank per source (- = not in top 50): " + "; ".join(ranks))

    print(f"Missed by every retriever at k=10 ({len(missed)}):")
    for q in missed:
        show(q)
    print(f"\nEvery retriever finds some but not all gold sources in the top 10 ({len(partial)}):")
    for q in partial:
        show(q)


def print_margins(runs: dict[str, dict[str, Any]], baseline: str, n: int = 10) -> None:
    dense = [name for name in runs if name in DENSE_NAMES]
    if baseline not in runs or not dense:
        return
    base = {i["qid"]: i for i in answerable(runs[baseline])}
    by = {name: {i["qid"]: i for i in answerable(runs[name])} for name in dense}

    def rank_str(item: dict[str, Any]) -> str:
        r = first_rank(item)
        return str(r) if r else "-"

    rows = []
    for q, b in base.items():
        best = max(dense, key=lambda name: rr(by[name][q]))
        worst = min(dense, key=lambda name: rr(by[name][q]))
        rows.append((q, b, rr(by[best][q]) - rr(b), best, rr(b) - rr(by[worst][q])))

    def show(q: str, b: dict[str, Any], margin: float) -> None:
        print(f"  {q} [{b['category']}] margin {margin:+.3f}  {b['question']}")
        print(f"    first relevant rank: {baseline} {rank_str(b)}; "
              + "; ".join(f"{name} {rank_str(by[name][q])}" for name in dense))

    print(f"Dense beats {baseline} by the largest margin (best dense model's 1/rank minus {baseline}'s; "
          "first relevant chunk, top 50):")
    for q, b, margin, _, _ in sorted(rows, key=lambda r: (-r[2], r[0]))[:n]:
        if margin > 0:
            show(q, b, margin)
    print(f"\n{baseline} beats every dense model by the largest margin ({baseline}'s 1/rank minus the best dense):")
    reverse = sorted(((q, b, rr(b) - max(rr(by[name][q]) for name in dense)) for q, b in base.items()),
                     key=lambda r: (-r[2], r[0]))
    for q, b, margin in reverse[:5]:
        if margin > 0:
            show(q, b, margin)


def report(runs: dict[str, dict[str, Any]], baseline: str) -> None:
    first = next(iter(runs.values()))
    print(f"Questions: {first['questions']} (sha256 {first['questions_sha256'][:12]}), {first['n_items']} items, "
          f"{first['n_answerable']} answerable. Commits: "
          + ", ".join(sorted({r['git_commit'] for r in runs.values()})))
    print("Metrics: " + first["metrics_definition"] + ". CIs: 95% percentile bootstrap, 1,000 resamples, seed 42.\n")
    print("Overall")
    print_overall(runs)
    print("\nBy collection")
    print_grouped(runs, "collection", "collection")
    print("\nBy category")
    print_grouped(runs, "category", "category")
    if baseline in runs:
        print(f"\nPaired bootstrap vs {baseline} (same items, same resamples; * = CI excludes 0)")
        print_paired(runs, baseline)
    print()
    print_misses(runs)
    print()
    print_margins(runs, baseline)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retrievers", nargs="+", default=None,
                    help="default: every registered retriever with a results file")
    ap.add_argument("--baseline", default="bm25", choices=BM25_NAMES + DENSE_NAMES)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--save", type=Path, default=None, help="also write the report to this file")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    names = args.retrievers or [n for n in RETRIEVER_NAMES if (args.results_dir / f"{n}.json").exists()]
    runs = load_results(names, args.results_dir)
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(runs, args.baseline)
    text = buf.getvalue()
    print(text, end="")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
