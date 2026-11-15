"""Run the answer-generation step on every eval item and score it automatically.

Usage::

    python scripts/run_generation_eval.py [--model mistral] [--pipeline v2] [--name mistral-v2]
                                          [--questions eval/questions.jsonl] [--out-dir results/generation]
                                          [--qids hd-007 aw-001] [--rescore [--out-name NAME]]

For each item it calls ``src/generate/answer.Answerer`` (bge-base top 5; v2: router, then refusal
gate; Ollama), checks citations and numbers (``src/generate/verify.py``) and writes one JSON line
to ``<out-dir>/<name>.jsonl`` (``<name>`` defaults to ``<model>-<pipeline>``): question, route,
retrieved chunk ids and scores, the sources shown to the model, gate score, every raw model
output (one per attempt), the parsed answer, citation checks, number flags and latency.
``<out-dir>/<name>.meta.json`` records the config, model digest, git commit and date.

It then prints the automatic metrics (``src/evaluate/generation_metrics.py``) overall and by
category, the personal-advice answers (for review of the redirect check), the dangerous answers,
and every failed item with its output; the same report goes to ``<out-dir>/<name>.report.txt``
and the numbers to ``<out-dir>/<name>.metrics.json``.

``--rescore`` recomputes checks and metrics from an existing ``<name>.jsonl`` without calling
the model, writing to ``--out-name`` if given (e.g. the v1 results re-scored with the current
checks: ``--rescore --name mistral-v1 --out-name mistral-v1-rescored``). ``--qids`` restricts the
run to some items (the files then hold only those). ``scripts/compare_generation.py`` puts
several runs side by side.

Long runs are resumable: each answered item is appended to ``<model>.partial.jsonl`` at once,
and running the same command again skips items already there (``--fresh`` starts over; a
different config or question file is refused). The run stops cleanly when free disk space on
the results or system drive drops below ``--min-free-gb`` (default 2), because Windows grows
its page file on the system drive when the LLM and this process use too much memory.
Retrieval for all items is done first in a child process that then exits, so the embedding
model is not held in memory while the LLM runs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate.eval_schema import EvalItem, parse_item  # noqa: E402
from src.evaluate.generation_metrics import aggregate, aggregate_by, as_dict, result_route, score_item  # noqa: E402
from src.evaluate.relevance import load_chunks, relevant_chunk_ids  # noqa: E402
from src.generate.config import PIPELINES, GenerationConfig  # noqa: E402
from src.generate.router import default_lexicon  # noqa: E402
from src.generate.verify import check_result  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
OUT_DIR = ROOT / "results" / "generation"


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model)


WARM_UP = "What is the first-line treatment for hypertension?"


def free_gb(paths: list[Path]) -> float:
    """Smallest free space (GB) over the drives of ``paths`` (results drive and system drive)."""
    return min(shutil.disk_usage(p).free for p in paths) / 1e9


def load_partial(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut short by a crash
                continue
            rows[row["qid"]] = row
    return rows


def run_items(items: list[EvalItem], config: GenerationConfig, chunks: dict[str, dict[str, Any]],
              partial: Path, min_free_gb: float) -> tuple[list[dict[str, Any]], bool]:
    """Answer every item, appending each row to ``partial`` as it is done; items already in
    ``partial`` are not run again. Stops early (returns ``complete=False``) if free disk space
    falls below ``min_free_gb``."""
    from concurrent.futures import ProcessPoolExecutor

    from src.generate.answer import Answerer, PrecomputedRetriever, retrieve_hits

    done = load_partial(partial)
    todo = [it for it in items if it.qid not in done]
    drives = [partial.parent, Path(os.environ.get("SystemDrive", "C:") + "\\")]
    if done:
        print(f"resuming: {len(items) - len(todo)} of {len(items)} items already in {partial.name}", flush=True)
    if todo and free_gb(drives) < 2 * min_free_gb:  # loading the model can briefly grow the page file by GBs
        print(f"STOPPED: free disk space {free_gb(drives):.1f} GB < {2 * min_free_gb} GB before loading the model.",
              flush=True)
        return [done[it.qid] for it in items if it.qid in done], False
    if todo:
        # Retrieval runs in a child process that exits before the LLM starts, so the embedding
        # model's memory (several GB with torch) is not held during generation.
        with ProcessPoolExecutor(max_workers=1) as pool:
            hits = pool.submit(retrieve_hits, [WARM_UP, *(it.question for it in todo)], config).result()
        answerer = Answerer(config, retriever=PrecomputedRetriever(config.retriever, hits), chunks=chunks)
        answerer.answer(WARM_UP)  # warm-up: loads the model
    complete = True
    with partial.open("a", encoding="utf-8") as fh:
        for item in todo:
            if free_gb(drives) < min_free_gb:
                print(f"STOPPED: free disk space {free_gb(drives):.1f} GB < {min_free_gb} GB. "
                      f"Progress is saved in {partial.name}; free space and run the same command to resume.",
                      flush=True)
                complete = False
                break
            result = answerer.answer(item.question)
            row = {"qid": item.qid, "category": item.category, "answerable": item.answerable,
                   "expected_behaviour": item.expected_behaviour, **result}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            done[item.qid] = row
            status = result["route"] + (" refused" if result["refused"] else ("" if result["valid"] else " INVALID"))
            print(f"[{len(done):>2}/{len(items)}] {item.qid:<8} {status:<12} "
                  f"{result['latency_ms']['total'] / 1000:6.1f} s  (disk free {free_gb(drives):.1f} GB)", flush=True)
    return [done[it.qid] for it in items if it.qid in done], complete


def add_checks(rows: list[dict[str, Any]], chunks: dict[str, dict[str, Any]]) -> None:
    for row in rows:
        row.update(check_result(row, chunks))


CITATION_MARK = {"verified": "ok ", "verified_wrong_label": "LBL", "not_verified": "BAD"}


def output_text(row: dict[str, Any]) -> str:
    route = result_route(row)
    if route == "gate":
        return f"[gate refusal, score {row['gate_score']:.3f} < {row['gate_threshold']}] {row['answer']}"
    if row.get("refused") and route != "llm" and route != "dose_lookup":
        detail = f" {row['route_detail']}" if row.get("route_detail") else ""
        return f"[route {route}{detail}] {row['answer']}"
    if row.get("refused") and route == "dose_lookup":
        return f"[dose lookup: no row for this weight; {row['route_detail']}] {row['answer']}"
    if not row["valid"]:
        return "[invalid output] raw: " + " || ".join(a["raw"][:400] for a in row["attempts"])
    lines = []
    if route == "dose_lookup":
        d = row["route_detail"]
        rows = "; ".join(f"{r['antibiotic']} {r['weight_band']}: {r['resolved_dose']} {r['resolved_frequency']}, "
                         f"{r['duration'][:40]}" for r in d["rows"])
        lines.append(f"[dose lookup {d['weight_kg']:g} kg, {d['infection']}: {rows}]")
    lines.append(row["answer"])
    for c in row.get("citation_checks", []):
        mark = CITATION_MARK.get(c.get("status") or ("verified" if c["verified"] else "not_verified"), "BAD")
        if c["verified"] and c.get("tolerant"):
            mark = "ok~"
        note = ""
        if c.get("status") == "verified_wrong_label":
            note = f"  (matches {c['correct_label']})"
        elif not c["verified"]:
            note = f"  ({c['reason']})"
        lines.append(f"    {mark} [{c['label']}] {c['chunk_id']}: \"{c['quote'][:160]}\"{note}")
    for s in row.get("unlabelled_number_sentences", []):
        lines.append(f"    no label on a sentence with a number: {s[:160]}")
    return "\n".join(lines)


def fmt(r: dict[str, Any]) -> str:
    return "-" if not r["of"] else f"{r['n']}/{r['of']} ({r['rate']:.0%})"


METRIC_ROWS = (
    ("route: personal-advice redirect", "route_redirect"),
    ("route: drug-grounding refusal", "route_drug_refusal"),
    ("route: weight-based dose lookup", "route_dose_lookup"),
    ("route: gate refusal", "route_gate"),
    ("route: LLM", "route_llm"),
    ("correct refusals (unanswerable)", "correct_refusals"),
    ("  of which redirect", "correct_refusals_redirect"),
    ("  of which drug grounding", "correct_refusals_drug_grounding"),
    ("  of which dose lookup (no row)", "correct_refusals_dose_no_match"),
    ("  of which gate", "correct_refusals_gate"),
    ("  of which LLM", "correct_refusals_llm"),
    ("false refusals (answerable)", "false_refusals"),
    ("  of which drug grounding", "false_refusals_drug_grounding"),
    ("  of which dose lookup (no row)", "false_refusals_dose_no_match"),
    ("  of which gate", "false_refusals_gate"),
    ("  of which LLM", "false_refusals_llm"),
    ("redirect to professional (personal_advice)", "redirect_ok"),
    ("citation accuracy (citations verified)", "citation_accuracy"),
    ("  verified only with edge tolerance", "citations_tolerant"),
    ("citations under the wrong label", "citations_wrong_label"),
    ("answers with all citations verified", "answers_all_verified"),
    ("answers without citations", "answers_without_citations"),
    ("answers with [Sn] labels in the text", "answers_with_labels_in_text"),
    ("answers: every number sentence labelled", "answers_numbers_labelled"),
    ('answers with inline "quote" [Sn] (v1 rule)', "answers_with_inline_quotes"),
    ("citation relevance (gold chunk cited, answerable)", "citation_relevance"),
    ("dose correct (gold_structured items)", "dose_correct"),
    ("answers with unsupported numbers", "answers_with_unsupported_numbers"),
    ("dangerous answers", "dangerous_answers"),
    ("invalid outputs (after retry)", "invalid_outputs"),
    ("items with any failure", "items_failed"),
)
CATEGORY_COLS = (("n", None), ("refused", None), ("false ref", "false_refusals"), ("correct ref", "correct_refusals"),
                 ("cit acc", "citation_accuracy"), ("all verif", "answers_all_verified"),
                 ("gold cited", "citation_relevance"), ("dose ok", "dose_correct"),
                 ("unsupp num", "answers_with_unsupported_numbers"), ("danger", "dangerous_answers"),
                 ("failed", "items_failed"))


def report(rows: list[dict[str, Any]], items: dict[str, EvalItem], scores: list[Any], meta: dict[str, Any]) -> None:
    overall = aggregate(scores)
    by_cat = aggregate_by(scores)
    cfg = meta["config"]
    print(f"run={meta.get('name', meta['model'])}  model={meta['model']}  pipeline={cfg.get('pipeline', 'v1')}  "
          f"scoring={meta.get('scoring', 'v2 checks')}  commit={meta['git_commit']}  items={len(rows)}  "
          f"gate threshold={cfg['gate_threshold']}  retriever={cfg['retriever']} top {cfg['top_k']}")
    lat = [r["latency_ms"]["total"] for r in rows if r.get("attempts")]
    if lat:
        print(f"latency (items sent to the LLM): median {statistics.median(lat) / 1000:.1f} s, "
              f"max {max(lat) / 1000:.1f} s; retries used: {sum(len(r['attempts']) > 1 for r in rows)}\n")
    w = max(len(label) for label, _ in METRIC_ROWS)
    print(f"{'metric':<{w}}  overall")
    print("-" * (w + 20))
    for label, key in METRIC_ROWS:
        print(f"{label:<{w}}  {fmt(overall[key])}")
    hd = next((s for s in scores if s.qid == "hd-007"), None)
    if hd is not None:
        label = '80 g check (hd-007 must not contain "80 g")'
        print(f"{label:<{w}}  {'FAIL' if hd.has_80g else 'pass'}")

    print("\nBy category")
    headers = ["category", *(c for c, _ in CATEGORY_COLS)]
    table = []
    for cat, agg in by_cat.items():
        cells = [cat]
        for col, key in CATEGORY_COLS:
            if col == "n":
                cells.append(str(agg["n_items"]))
            elif col == "refused":
                cells.append(str(sum(s.refused for s in scores if s.category == cat)))
            else:
                r = agg[key]
                cells.append("-" if not r["of"] else f"{r['n']}/{r['of']}")
        table.append(cells)
    widths = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(wd) if i == 0 else h.rjust(wd) for i, (h, wd) in enumerate(zip(headers, widths))))
    for t in table:
        print("  ".join(c.ljust(wd) if i == 0 else c.rjust(wd) for i, (c, wd) in enumerate(zip(t, widths))))

    by_qid = {r["qid"]: r for r in rows}
    print("\nPersonal-advice answers (redirect check; review these)")
    for s in scores:
        if s.category == "personal_advice":
            print(f"\n  {s.qid} redirect={'yes' if s.redirect_ok else 'NO'}  Q: {items[s.qid].question}")
            print("    " + output_text(by_qid[s.qid]).replace("\n", "\n    "))

    dangerous = [s for s in scores if s.dangerous]
    print(f"\nDangerous answers ({len(dangerous)} of {len(scores)})")
    for s in dangerous:
        print(f"\n  {s.qid} [{s.category}] {'; '.join(s.dangerous)}")
        print(f"    Q: {items[s.qid].question}")
        print("    A: " + output_text(by_qid[s.qid]).replace("\n", "\n    "))

    failed = [s for s in scores if s.failures]
    print(f"\nFailed items ({len(failed)} of {len(scores)})")
    for s in failed:
        r = by_qid[s.qid]
        print(f"\n  {s.qid} [{s.category}] route={s.route}: {'; '.join(s.failures)}")
        print(f"    Q: {items[s.qid].question}")
        print(f"    retrieved: {', '.join(x['label'] + '=' + x['chunk_id'] for x in r['retrieved'])} "
              f"(gate {r['gate_score']:.3f})")
        print("    A: " + output_text(r).replace("\n", "\n    "))


def main(argv: list[str] | None = None) -> int:
    defaults = GenerationConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=defaults.model)
    ap.add_argument("--pipeline", choices=PIPELINES, default=defaults.pipeline)
    ap.add_argument("--name", default=None, help="run name for the files (default <model>-<pipeline>)")
    ap.add_argument("--out-name", default=None, help="with --rescore: write the re-scored files under this name")
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--qids", nargs="+", default=None)
    ap.add_argument("--rescore", action="store_true", help="re-score an existing results file without the LLM")
    ap.add_argument("--num-gpu", type=int, default=None, help="layers offloaded to the GPU (0 = CPU only)")
    ap.add_argument("--min-free-gb", type=float, default=2.0,
                    help="stop (resumably) when free disk space drops below this many GB")
    ap.add_argument("--fresh", action="store_true", help="discard saved progress and start again")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = defaults.with_(model=args.model, num_gpu=args.num_gpu, pipeline=args.pipeline)
    all_items = [parse_item(json.loads(line)) for line in args.questions.read_text(encoding="utf-8").splitlines() if line]
    items = [it for it in all_items if args.qids is None or it.qid in args.qids]
    records = load_chunks()
    chunks = {c["chunk_id"]: c for c in records}
    name = args.name or f"{model_slug(args.model)}-{args.pipeline}"
    out_name = args.out_name or name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl, meta_path = args.out_dir / f"{name}.jsonl", args.out_dir / f"{name}.meta.json"

    if args.rescore:
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if out_name != name:
            meta = {**meta, "rescored_from": jsonl.name}
    else:
        from src.generate.llm import OllamaClient

        partial, partial_meta = args.out_dir / f"{name}.partial.jsonl", args.out_dir / f"{name}.partial.meta.json"
        questions_sha = hashlib.sha256(args.questions.read_bytes()).hexdigest()
        run_key = {"config": config.as_dict(), "questions_sha256": questions_sha}
        if args.fresh:
            partial.unlink(missing_ok=True)
        if partial.exists() and partial_meta.exists():
            saved = json.loads(partial_meta.read_text(encoding="utf-8"))
            if saved != run_key:
                print(f"{partial.name} was made with a different config or question file; "
                      "use --fresh to discard it.", file=sys.stderr)
                return 1
        partial_meta.write_text(json.dumps(run_key, indent=1), encoding="utf-8")
        t0 = time.perf_counter()
        rows, complete = run_items(items, config, chunks, partial, args.min_free_gb)
        if not complete:
            return 2
        meta = {
            "name": name,
            "model": args.model,
            "model_info": OllamaClient(args.model, config.ollama_url).model_info(),
            "config": config.as_dict(),
            "git_commit": git_commit(),
            "date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "questions": args.questions.as_posix(),
            "questions_sha256": questions_sha,
            "n_items": len(rows),
            "wall_time_s": round(time.perf_counter() - t0, 1),  # this invocation only (if resumed)
            "retrieval": "precomputed in a child process (same retriever and config)",
        }
    meta = {**meta, "name": out_name, "scoring": "v2 checks"}
    add_checks(rows, chunks)
    out_jsonl, out_meta = args.out_dir / f"{out_name}.jsonl", args.out_dir / f"{out_name}.meta.json"
    out_jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out_meta.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if not args.rescore:  # the final files now hold everything
        partial.unlink(missing_ok=True)
        partial_meta.unlink(missing_ok=True)

    item_by_qid = {it.qid: it for it in all_items}
    lexicon = default_lexicon()
    scores = [score_item(item_by_qid[r["qid"]], r, r, relevant_chunk_ids(item_by_qid[r["qid"]], records),
                         chunks=chunks, lexicon=lexicon) for r in rows]
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(rows, item_by_qid, scores, meta)
    text = buf.getvalue()
    print(text, end="")
    (args.out_dir / f"{out_name}.report.txt").write_text(text, encoding="utf-8")
    (args.out_dir / f"{out_name}.metrics.json").write_text(json.dumps({
        "name": out_name, "model": meta["model"], "pipeline": meta["config"].get("pipeline", "v1"),
        "scoring": meta["scoring"], "overall": aggregate(scores), "by_category": aggregate_by(scores),
        "items": [as_dict(s) for s in scores]}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
