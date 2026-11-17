"""Side-by-side table of generation runs (``results/generation/<name>.metrics.json``).

Usage::

    python scripts/compare_generation.py [mistral-v1-rescored mistral-v2 mistral-v2.1 qwen2.5-7b-v2]

Writes the table to stdout and to ``results/generation/comparison.txt``. A run's metrics file
lacks the keys that its scoring version did not compute (the original v1 file has no route,
wrong-label or dangerous-answer counts); those cells show "n/a".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "generation"

ROWS = (
    ("route: personal-advice redirect", "route_redirect"),
    ("route: drug-grounding refusal", "route_drug_refusal"),
    ("route: weight-based dose lookup", "route_dose_lookup"),
    ("route: gate refusal", "route_gate"),
    ("route: LLM", "route_llm"),
    ("correct refusals (unanswerable)", "correct_refusals"),
    ("false refusals (answerable)", "false_refusals"),
    ("  gate", "false_refusals_gate"),
    ("  LLM", "false_refusals_llm"),
    ("  dose lookup, no row", "false_refusals_dose_no_match"),
    ("  drug grounding", "false_refusals_drug_grounding"),
    ("redirect to professional (personal advice)", "redirect_ok"),
    ("citation accuracy", "citation_accuracy"),
    ("  of which needed edge tolerance", "citations_tolerant"),
    ("citations under the wrong label", "citations_wrong_label"),
    ("answers with all citations verified", "answers_all_verified"),
    ("answers with [Sn] labels in text", "answers_with_labels_in_text"),
    ("answers: every number sentence labelled", "answers_numbers_labelled"),
    ("citation relevance (gold chunk cited)", "citation_relevance"),
    ("dose correct", "dose_correct"),
    ("answers with unsupported numbers", "answers_with_unsupported_numbers"),
    ("dangerous answers", "dangerous_answers"),
    ("invalid outputs", "invalid_outputs"),
    ("items with any failure", "items_failed"),
)


def cell(r: dict | None) -> str:
    if r is None:
        return "n/a"
    if not r["of"]:
        return "-"
    return f"{r['n']}/{r['of']} ({r['rate']:.0%})"


def main(argv: list[str]) -> int:
    names = argv or ["mistral-v1-rescored", "mistral-v2", "mistral-v2.1", "qwen2.5-7b-v2"]
    runs = []
    for n in names:
        path = OUT_DIR / f"{n}.metrics.json"
        if path.exists():
            runs.append((n, json.loads(path.read_text(encoding="utf-8"))))
        else:
            print(f"(skipping {n}: no {path.name})", file=sys.stderr)
    w = max(len(label) for label, _ in ROWS)
    cols = [[cell(m["overall"].get(key)) for _, key in ROWS] for _, m in runs]
    widths = [max(len(n), *(len(c) for c in col)) for (n, _), col in zip(runs, cols)]
    lines = ["  ".join([f"{'metric':<{w}}", *(n.rjust(wd) for (n, _), wd in zip(runs, widths))])]
    lines.append("-" * len(lines[0]))
    for i, (label, _) in enumerate(ROWS):
        lines.append("  ".join([f"{label:<{w}}", *(col[i].rjust(wd) for col, wd in zip(cols, widths))]))
    hd = []
    for _, m in runs:
        s = next((x for x in m["items"] if x["qid"] == "hd-007"), None)
        hd.append("n/a" if s is None else ("FAIL" if s.get("has_80g") else "pass"))
    lines.append("  ".join([f"{'hd-007 80 g check':<{w}}", *(h.rjust(wd) for h, wd in zip(hd, widths))]))
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (OUT_DIR / "comparison.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
