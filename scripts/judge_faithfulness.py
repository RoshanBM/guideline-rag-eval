"""LLM judge of answer faithfulness, and its agreement with the human labels.

Usage::

    python scripts/judge_faithfulness.py judge [--sheet notes/faithfulness_labels.csv] [--model qwen2.5:7b]
                                               [--out-dir results/faithfulness] [--fresh]
    python scripts/judge_faithfulness.py agree [--sheet notes/faithfulness_labels.csv]
                                               [--judgements results/faithfulness/qwen2.5-7b.jsonl] [--use overall]

**judge**: for every row of the labelling sheet (``scripts/make_faithfulness_sheet.py``) the judge
model gets the question, the answer and the full text of each cited source (the same text the human
labeller sees; the generator's quotes are not shown). It splits the answer into claims, labels each
``supported`` / ``unsupported`` / ``contradicted`` and gives an overall ``faithful`` / ``partial`` /
``no``, following ``notes/faithfulness_labelling_guide.md``. The judge is ``qwen2.5:7b`` via Ollama,
a different model from the generator (mistral), at temperature 0, seed 42. Output is JSON (Ollama
structured output), validated, with one retry on invalid output. One JSON line per item goes to
``<out-dir>/<model>.jsonl``, written as each item finishes (a rerun skips finished items unless
``--fresh``). It holds every raw reply, the claims, the judge's overall label, the overall label
implied by the claim labels (``derived_overall``) and whether the two agree. Run settings go to
``<model>.meta.json``.

**agree**: compares the human ``faithful`` column (yes / partial / no) with the judge's overall
label (faithful -> yes; ``--use derived`` takes the label implied by the claims). It reports
percent agreement, Cohen's kappa unweighted and linear-weighted (order no < partial < yes), each
with a 95 % percentile bootstrap CI over items (1,000 resamples, seed 42), and the confusion
matrix. It writes ``<out-dir>/agreement_<model>.json`` and ``.txt`` (``agreement_<model>_derived.*`` with
``--use derived``). Rows with no human label are
skipped; an unknown label is an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate.llm import OllamaClient, OllamaError  # noqa: E402

SHEET = ROOT / "notes" / "faithfulness_labels.csv"
OUT_DIR = ROOT / "results" / "faithfulness"
JUDGE_MODEL = "qwen2.5:7b"
OPTIONS = {"temperature": 0.0, "seed": 42, "num_ctx": 12288, "num_predict": 2048}
MAX_RETRIES = 1
N_RESAMPLES, SEED = 1000, 42

CLAIM_LABELS = ("supported", "unsupported", "contradicted")
OVERALL_LABELS = ("faithful", "partial", "no")
HUMAN_LABELS = ("no", "partial", "yes")  # ordinal order for the weighted kappa
TO_HUMAN = {"faithful": "yes", "partial": "partial", "no": "no"}

SYSTEM_PROMPT = """You check whether an answer about WHO clinical guidelines is faithful to the sources it cites.

Use ONLY the numbered sources in the user message. Do not use your own medical knowledge, and do not judge whether the answer is clinically correct: only whether the sources back it.

Steps:
1. Split the answer into claims. A claim is one statement of fact or recommendation (including numbers, drugs, doses, populations, conditions and strength of recommendation). Ignore the [S1]-style labels when splitting, and do not treat them as claims.
2. Label each claim:
   - "supported": a source states it, or it follows directly from what a source states (a reasonable paraphrase counts; a broader generalisation does not). A claim is supported if ANY source supports it, whatever label the answer attached to it.
   - "unsupported": no source states it, but nothing in the sources contradicts it.
   - "contradicted": a source says otherwise, or the claim invents a specific detail that no source gives (for example a dose, drug, number, threshold or population).
3. Give the overall label:
   - "faithful": every claim is supported.
   - "partial": at least one claim is unsupported, and no claim is contradicted.
   - "no": at least one claim is contradicted.

Reply with only a JSON object of this form:
{"claims": [{"claim": "<the claim>", "label": "supported", "sources": ["S1"], "reason": "<short reason>"}], "overall": "faithful"}
For an unsupported or contradicted claim, "sources" may be empty."""

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "label": {"type": "string", "enum": list(CLAIM_LABELS)},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "label", "sources", "reason"],
            },
        },
        "overall": {"type": "string", "enum": list(OVERALL_LABELS)},
    },
    "required": ["claims", "overall"],
}


# --------------------------------------------------------------------------- #
# Sheet
# --------------------------------------------------------------------------- #
def read_sheet(path: Path) -> list[dict[str, Any]]:
    """Rows of the labelling sheet: item_id, question, answer, sources [{label, text}] (one per label)."""
    out = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            sources: dict[str, str] = {}
            i = 1
            while f"source_{i}_label" in row:
                label = (row[f"source_{i}_label"] or "").strip()
                if label and label not in sources:
                    sources[label] = row[f"source_{i}_text"] or ""
                i += 1
            out.append({"item_id": row["item_id"], "question": row["question"], "answer": row["answer"],
                        "sources": [{"label": k, "text": v} for k, v in sorted(sources.items())],
                        "faithful": (row.get("faithful") or "").strip().lower()})
    return out


def build_user_prompt(item: dict[str, Any]) -> str:
    blocks = "\n\n".join(f"[{s['label']}]\n{s['text']}" for s in item["sources"])
    labels = ", ".join(s["label"] for s in item["sources"])
    return (f"Question: {item['question']}\n\nAnswer to check:\n{item['answer']}\n\n"
            f"Sources ({labels}):\n\n{blocks}\n\nLabel every claim of the answer and give the overall label.")


# --------------------------------------------------------------------------- #
# Judge output
# --------------------------------------------------------------------------- #
def derived_overall(claim_labels: list[str]) -> str:
    """The overall label the claim labels imply (the guide's rule)."""
    if "contradicted" in claim_labels:
        return "no"
    if "unsupported" in claim_labels:
        return "partial"
    return "faithful"


def parse_judgement(raw: str, labels: list[str]) -> tuple[dict[str, Any] | None, list[str]]:
    """``(parsed, errors)``; ``parsed`` is None if the reply is invalid."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON ({exc.msg})"]
    if not isinstance(obj, dict):
        return None, ["output is not a JSON object"]
    errors: list[str] = []
    claims, overall = obj.get("claims"), obj.get("overall")
    if overall not in OVERALL_LABELS:
        errors.append(f'"overall" must be one of {", ".join(OVERALL_LABELS)}')
    if not isinstance(claims, list) or not claims:
        errors.append('"claims" must be a non-empty list')
        claims = []
    clean = []
    for i, c in enumerate(claims):
        if not isinstance(c, dict) or not isinstance(c.get("claim"), str) or not c["claim"].strip():
            errors.append(f"claims[{i}] needs a non-empty \"claim\"")
            continue
        if c.get("label") not in CLAIM_LABELS:
            errors.append(f"claims[{i}] label must be one of {', '.join(CLAIM_LABELS)}")
            continue
        srcs = c.get("sources") if isinstance(c.get("sources"), list) else []
        srcs = [str(s).strip().strip("[]").upper() for s in srcs]
        unknown = [s for s in srcs if s not in labels]
        if unknown:
            errors.append(f"claims[{i}] cites unknown sources {unknown} (use {', '.join(labels)})")
        clean.append({"claim": c["claim"].strip(), "label": c["label"], "sources": srcs,
                      "reason": str(c.get("reason") or "")})
    if errors:
        return None, errors
    derived = derived_overall([c["label"] for c in clean])
    return {"claims": clean, "overall": overall, "derived_overall": derived, "consistent": derived == overall}, []


def judge_item(client: Any, item: dict[str, Any]) -> dict[str, Any]:
    labels = [s["label"] for s in item["sources"]]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_user_prompt(item)}]
    attempts, parsed = [], None
    t0 = time.perf_counter()
    for _ in range(1 + MAX_RETRIES):
        t1 = time.perf_counter()
        try:
            reply = client.chat(messages, format=OUTPUT_SCHEMA)
        except (OllamaError, OSError) as exc:  # includes timeouts: counts as an invalid attempt
            attempts.append({"raw": "", "errors": [f"LLM call failed: {exc!r}"],
                             "ms": round((time.perf_counter() - t1) * 1000, 1), "stats": {}})
            continue
        parsed, errors = parse_judgement(reply.content, labels)
        stats = reply.stats or {}
        ctx_full = (stats.get("prompt_eval_count") or 0) + (stats.get("eval_count") or 0) >= OPTIONS["num_ctx"]
        attempts.append({"raw": reply.content, "errors": errors, "ms": round((time.perf_counter() - t1) * 1000, 1),
                         "stats": stats, "context_full": ctx_full})
        if parsed is not None:
            break
        messages += [{"role": "assistant", "content": reply.content},
                     {"role": "user", "content": "Your reply was invalid: " + "; ".join(errors)
                      + ". Reply again with only the corrected JSON object."}]
    return {"item_id": item["item_id"], "valid": parsed is not None, **(parsed or {"claims": [], "overall": None,
            "derived_overall": None, "consistent": None}), "attempts": attempts,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def confusion(human: list[str], judge: list[str], cats: tuple[str, ...] = HUMAN_LABELS) -> list[list[int]]:
    """Counts with rows = human label, columns = judge label, both in ``cats`` order."""
    idx = {c: i for i, c in enumerate(cats)}
    m = [[0] * len(cats) for _ in cats]
    for h, j in zip(human, judge):
        m[idx[h]][idx[j]] += 1
    return m


def cohen_kappa(human: list[str], judge: list[str], weights: str | None = None,
                cats: tuple[str, ...] = HUMAN_LABELS) -> float | None:
    """Cohen's kappa; ``weights="linear"`` for the linear-weighted kappa over the ordered ``cats``.
    None when chance disagreement is 0 (e.g. both raters use a single category)."""
    n, k = len(human), len(cats)
    if n == 0:
        return None
    m = confusion(human, judge, cats)
    rows = [sum(r) for r in m]
    cols = [sum(m[i][j] for i in range(k)) for j in range(k)]
    if weights == "linear":
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    observed = sum(w[i][j] * m[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w[i][j] * rows[i] * cols[j] for i in range(k) for j in range(k)) / (n * n)
    return None if expected == 0 else 1.0 - observed / expected


def bootstrap_ci(human: list[str], judge: list[str], stat: Any, n_resamples: int = N_RESAMPLES,
                 seed: int = SEED) -> tuple[list[float | None], int]:
    """95 % percentile CI of ``stat(human, judge)`` over items; also the number of resamples where
    the statistic is undefined (they are left out)."""
    rng = random.Random(seed)
    n, values, undefined = len(human), [], 0
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        v = stat([human[i] for i in idx], [judge[i] for i in idx])
        if v is None:
            undefined += 1
        else:
            values.append(v)
    if not values:
        return [None, None], undefined
    values.sort()
    lo = values[int(0.025 * (len(values) - 1))]
    hi = values[int(round(0.975 * (len(values) - 1)))]
    return [round(lo, 4), round(hi, 4)], undefined


def agreement(human: list[str], judge: list[str]) -> dict[str, Any]:
    n = len(human)
    pct = sum(h == j for h, j in zip(human, judge)) / n if n else None
    out: dict[str, Any] = {"n": n, "percent_agreement": None if pct is None else round(pct, 4),
                           "categories": list(HUMAN_LABELS),
                           "confusion_rows_human_cols_judge": confusion(human, judge)}
    for name, w in (("kappa", None), ("kappa_linear", "linear")):
        def stat(h: list[str], j: list[str], w: str | None = w) -> float | None:
            return cohen_kappa(h, j, w)
        k = stat(human, judge)
        ci, undefined = bootstrap_ci(human, judge, stat) if n else ([None, None], 0)
        out[name] = {"value": None if k is None else round(k, 4), "ci95": ci, "undefined_resamples": undefined}
    out["percent_agreement_ci95"] = (bootstrap_ci(human, judge, percent_agreement)[0] if n else [None, None])
    return out


def percent_agreement(human: list[str], judge: list[str]) -> float:
    return sum(h == j for h, j in zip(human, judge)) / len(human)


def agreement_text(a: dict[str, Any], name: str) -> str:
    def k(x: dict[str, Any]) -> str:
        return "undefined" if x["value"] is None else f"{x['value']:.3f} [{x['ci95'][0]}, {x['ci95'][1]}]"
    lines = [f"Judge {name} vs human 'faithful' labels, n = {a['n']}",
             f"percent agreement      {a['percent_agreement']:.3f} {a['percent_agreement_ci95']}" if a["n"] else "no items",
             f"Cohen's kappa          {k(a['kappa'])}",
             f"linear-weighted kappa  {k(a['kappa_linear'])}",
             "(95 % percentile bootstrap over items, 1,000 resamples, seed 42)", "",
             "confusion (rows = human, columns = judge)",
             "          " + "".join(f"{c:>9}" for c in a["categories"])]
    for c, row in zip(a["categories"], a["confusion_rows_human_cols_judge"]):
        lines.append(f"{c:<10}" + "".join(f"{v:>9}" for v in row))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def cmd_judge(args: argparse.Namespace) -> int:
    items = read_sheet(args.sheet)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{slug(args.model)}.jsonl"
    if args.fresh:
        out.unlink(missing_ok=True)
    done = {json.loads(line)["item_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()} \
        if out.exists() else set()
    client = OllamaClient(args.model, args.ollama_url, options=OPTIONS, timeout_s=600.0)
    meta = {"judge_model": args.model, "model_info": client.model_info(), "options": OPTIONS,
            "max_retries": MAX_RETRIES, "system_prompt": SYSTEM_PROMPT, "sheet": args.sheet.relative_to(ROOT).as_posix(),
            "sheet_sha256": hashlib.sha256(args.sheet.read_bytes()).hexdigest(), "git_commit": git_commit(),
            "date": datetime.now().astimezone().isoformat(timespec="seconds"), "n_items": len(items)}
    (args.out_dir / f"{slug(args.model)}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    with out.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(items, start=1):
            if item["item_id"] in done:
                continue
            r = judge_item(client, item)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            status = r["overall"] if r["valid"] else "INVALID"
            print(f"[{n:>2}/{len(items)}] {item['item_id']:<8} {status:<9} "
                  f"{'' if r['consistent'] in (True, None) else '(claims imply ' + str(r['derived_overall']) + ') '}"
                  f"{r['latency_ms'] / 1000:5.1f} s", flush=True)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def cmd_agree(args: argparse.Namespace) -> int:
    items = read_sheet(args.sheet)
    bad = [(it["item_id"], it["faithful"]) for it in items if it["faithful"] and it["faithful"] not in HUMAN_LABELS]
    if bad:
        print(f"unknown human labels (use {', '.join(HUMAN_LABELS)}): {bad}", file=sys.stderr)
        return 1
    judgements = {}
    for line in args.judgements.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            judgements[r["item_id"]] = r
    key = "derived_overall" if args.use == "derived" else "overall"
    pairs, skipped = [], {"no human label": [], "no valid judgement": []}
    for it in items:
        j = judgements.get(it["item_id"])
        if not it["faithful"]:
            skipped["no human label"].append(it["item_id"])
        elif not j or not j.get("valid") or j.get(key) is None:
            skipped["no valid judgement"].append(it["item_id"])
        else:
            pairs.append((it["item_id"], it["faithful"], TO_HUMAN[j[key]]))
    human, judge = [p[1] for p in pairs], [p[2] for p in pairs]
    result = {"judgements": args.judgements.name, "use": key, **agreement(human, judge), "skipped": skipped,
              "disagreements": [{"item_id": i, "human": h, "judge": j} for i, h, j in pairs if h != j]}
    name = args.judgements.stem
    text = agreement_text(result, f"{name} ({key})")
    text += "".join(f"\nskipped ({why}): {', '.join(ids)}" for why, ids in skipped.items() if ids)
    text += "".join(f"\ndisagree {d['item_id']}: human {d['human']}, judge {d['judge']}" for d in result["disagreements"])
    print(text)
    args.judgements.parent.mkdir(parents=True, exist_ok=True)
    stem = f"agreement_{name}" + ("_derived" if args.use == "derived" else "")
    (args.judgements.parent / f"{stem}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (args.judgements.parent / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    j = sub.add_parser("judge", help="run the LLM judge over the sheet")
    j.add_argument("--sheet", type=Path, default=SHEET)
    j.add_argument("--model", default=JUDGE_MODEL)
    j.add_argument("--ollama-url", default="http://localhost:11434")
    j.add_argument("--out-dir", type=Path, default=OUT_DIR)
    j.add_argument("--fresh", action="store_true", help="discard earlier judgements and start again")
    a = sub.add_parser("agree", help="agreement between the judge and the human labels")
    a.add_argument("--sheet", type=Path, default=SHEET)
    a.add_argument("--judgements", type=Path, default=OUT_DIR / f"{slug(JUDGE_MODEL)}.jsonl")
    a.add_argument("--use", choices=("overall", "derived"), default="overall",
                   help="judge label: its overall label, or the one its claim labels imply")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.sheet = args.sheet.resolve()
    return cmd_judge(args) if args.command == "judge" else cmd_agree(args)


if __name__ == "__main__":
    sys.exit(main())
