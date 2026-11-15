"""Top-1 retrieval score as a refusal signal, per retriever.

Usage::

    python scripts/refusal_analysis.py [--retrievers ...] [--results-dir results/retrieval]
                                       [--max-false-refusal 0.05]

For every ``results/retrieval/<name>.json`` it takes each item's top-1 score and reports:

* AUROC for separating answerable from unanswerable items (higher score = answerable), with a
  stratified 95 % bootstrap CI (``src/evaluate/refusal.py``);
* the threshold that refuses as many unanswerable items as possible while refusing at most 5 %
  of answerable ones (refuse when top-1 < threshold), with the resulting false-refusal rate
  (answerable refused) and refusal rate (unanswerable refused). The threshold is chosen on the
  eval set itself, so the rates are optimistic for new questions.

Scores are on each retriever's own scale (BM25 score, cosine, RRF, min-max fusion, reranker
logit), so thresholds do not transfer between retrievers; AUROC does compare.

Writes ``refusal.json``, ``refusal.txt`` (the printed table) and ``refusal_scores.png`` (top-1
score distributions, one panel per retriever) to the results directory.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.refusal import auroc_ci, threshold_at_max_false_refusal  # noqa: E402
from src.retrieve.registry import RERANK_NAMES, RETRIEVER_NAMES  # noqa: E402

RESULTS_DIR = ROOT / "results" / "retrieval"

# Chart tokens (dataviz reference palette, light mode; slots 1-2 validated with
# validate_palette.js: all checks pass, CVD dE 24.7).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"answerable": "#2a78d6", "unanswerable": "#eb6834"}


def load(names: list[str], results_dir: Path) -> dict[str, dict[str, Any]]:
    runs = {n: json.loads((results_dir / f"{n}.json").read_text(encoding="utf-8")) for n in names}
    if len({r["questions_sha256"] for r in runs.values()}) != 1:
        raise SystemExit("results come from different question files")
    return runs


def split_scores(run: dict[str, Any]) -> tuple[list[float | None], list[float | None]]:
    pos = [i["top_score"] for i in run["items"] if i["answerable"]]
    neg = [i["top_score"] for i in run["items"] if not i["answerable"]]
    return pos, neg


def analyse(runs: dict[str, dict[str, Any]], max_frr: float) -> dict[str, dict[str, Any]]:
    out = {}
    for name, run in runs.items():
        pos, neg = split_scores(run)
        ci = auroc_ci(pos, neg)
        th = threshold_at_max_false_refusal(pos, neg, max_frr)
        finite = lambda xs: [x for x in xs if x is not None]  # noqa: E731
        out[name] = {
            "n_answerable": len(pos), "n_unanswerable": len(neg),
            "auroc": round(ci.estimate, 4), "auroc_ci95": [round(ci.lo, 4), round(ci.hi, 4)],
            "median_top_answerable": statistics.median(finite(pos)),
            "median_top_unanswerable": statistics.median(finite(neg)),
            "n_missing_scores": sum(x is None for x in pos + neg),
            "threshold": th.threshold, "max_false_refusal": max_frr,
            "false_refusal_rate": round(th.false_refusal_rate, 4), "n_false_refusals": th.n_false_refusals,
            "refusal_rate_unanswerable": round(th.refusal_rate, 4),
            "n_refused_unanswerable": th.n_refused_unanswerable,
            "is_reranker": name in RERANK_NAMES,
        }
    return out


def print_table(stats: dict[str, dict[str, Any]], max_frr: float) -> None:
    headers = ["retriever", "AUROC [95% CI]", "median top-1 ans", "median top-1 unans",
               f"threshold (FRR<={max_frr:.0%})", "false refusals", "unanswerable refused"]
    rows = []
    for name, s in stats.items():
        rows.append([
            name, f"{s['auroc']:.3f} [{s['auroc_ci95'][0]:.2f}, {s['auroc_ci95'][1]:.2f}]",
            f"{s['median_top_answerable']:.4g}", f"{s['median_top_unanswerable']:.4g}",
            f"{s['threshold']:.4g}",
            f"{s['n_false_refusals']}/{s['n_answerable']} ({s['false_refusal_rate']:.1%})",
            f"{s['n_refused_unanswerable']}/{s['n_unanswerable']} ({s['refusal_rate_unanswerable']:.1%})",
        ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w)  # noqa: E731
                                   for i, (c, w) in enumerate(zip(cells, widths)))
    print(line(headers))
    print("-" * len(line(headers)))
    for r in rows:
        print(line(r))


def plot(runs: dict[str, dict[str, Any]], stats: dict[str, dict[str, Any]], path: Path) -> None:
    names = list(runs)
    ncols = 4
    nrows = math.ceil(len(names) / ncols)
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "text.color": INK,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.0 * nrows + 0.9), facecolor=SURFACE)
    rng = np.random.default_rng(0)  # jitter only
    for ax, name in zip(axes.flat, names):
        pos, neg = split_scores(runs[name])
        ax.set_facecolor(SURFACE)
        finite = [x for x in pos + neg if x is not None]
        lo = min(finite)
        for row, (label, scores) in enumerate((("unanswerable", neg), ("answerable", pos))):
            xs = np.array([lo if s is None else s for s in scores], dtype=float)
            ys = row + rng.uniform(-0.18, 0.18, len(xs))
            ax.scatter(xs, ys, s=30, color=COLOR[label], edgecolors=SURFACE, linewidths=1.0, zorder=3)
        s = stats[name]
        if s["is_reranker"] and math.isfinite(s["threshold"]):
            ax.axvline(s["threshold"], color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=2)
            ax.annotate(f"5% FRR threshold\nrefuses {s['n_refused_unanswerable']}/{s['n_unanswerable']}",
                        xy=(s["threshold"], 1.45), fontsize=7, color=INK_2, ha="left", va="center",
                        xytext=(4, 0), textcoords="offset points")
        ax.set_yticks([0, 1], [f"unanswerable ({len(neg)})", f"answerable ({len(pos)})"])
        ax.set_ylim(-0.5, 1.75)
        ax.set_title(f"{name}   AUROC {s['auroc']:.2f} [{s['auroc_ci95'][0]:.2f}, {s['auroc_ci95'][1]:.2f}]",
                     fontsize=9, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle("Top-1 retrieval score: answerable vs unanswerable items (each panel on its own score scale)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Higher top-1 score should mean answerable. AUROC 0.5 = no separation. "
             "Dashed line (rerankers): refuse below it; at most 5% of answerable items refused, "
             "chosen on this eval set.", fontsize=8, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--retrievers", nargs="+", default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--max-false-refusal", type=float, default=0.05)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    names = args.retrievers or [n for n in RETRIEVER_NAMES if (args.results_dir / f"{n}.json").exists()]
    runs = load(names, args.results_dir)
    stats = analyse(runs, args.max_false_refusal)
    buf = io.StringIO()
    with redirect_stdout(buf):
        first = next(iter(runs.values()))
        print(f"Refusal signal: top-1 score, {first['n_answerable']} answerable vs {first['n_unanswerable']} "
              f"unanswerable items. AUROC CI: stratified bootstrap, 1,000 resamples, seed 42. "
              f"Threshold chosen on this eval set (optimistic).\n")
        print_table(stats, args.max_false_refusal)
    text = buf.getvalue()
    print(text, end="")
    (args.results_dir / "refusal.txt").write_text(text, encoding="utf-8")
    (args.results_dir / "refusal.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    plot(runs, stats, args.results_dir / "refusal_scores.png")
    print(f"\nwrote refusal.txt, refusal.json and refusal_scores.png to {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
