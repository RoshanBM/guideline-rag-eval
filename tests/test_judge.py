"""scripts/judge_faithfulness.py: output parsing, the agreement statistics and the agree command
(no LLM calls), and scripts/make_faithfulness_sheet.py's allocation."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


judge = load("judge_faithfulness")
sheet = load("make_faithfulness_sheet")


# --------------------------------------------------------------------------- #
# Kappa and agreement
# --------------------------------------------------------------------------- #
def test_kappa_textbook_example():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: po = 0.70, pe = 0.50, kappa = 0.40
    human = ["yes"] * 25 + ["no"] * 25
    rater = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert judge.cohen_kappa(human, rater) == pytest.approx(0.4)
    assert judge.cohen_kappa(human, rater, "linear") == pytest.approx(0.4)  # all disagreements at full distance
    assert judge.confusion(human, rater) == [[15, 0, 10], [0, 0, 0], [5, 0, 20]]


def test_kappa_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    human = ["yes", "yes", "partial", "no", "partial", "yes", "no", "yes", "partial", "yes", "no", "yes"]
    rater = ["yes", "partial", "partial", "no", "yes", "yes", "partial", "yes", "no", "yes", "no", "partial"]
    labels = list(judge.HUMAN_LABELS)
    assert judge.cohen_kappa(human, rater) == pytest.approx(sk.cohen_kappa_score(human, rater, labels=labels))
    assert judge.cohen_kappa(human, rater, "linear") == pytest.approx(
        sk.cohen_kappa_score(human, rater, labels=labels, weights="linear"))


def test_kappa_edge_cases():
    assert judge.cohen_kappa(["yes"] * 4, ["yes"] * 4) is None  # chance agreement 1: undefined
    assert judge.cohen_kappa([], []) is None
    assert judge.cohen_kappa(["yes", "no", "partial"], ["yes", "no", "partial"]) == pytest.approx(1.0)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    human = ["yes", "yes", "partial", "no", "partial", "yes", "no", "yes", "partial", "yes"] * 2
    rater = ["yes", "partial", "partial", "no", "yes", "yes", "partial", "yes", "no", "yes"] * 2
    ci1, und1 = judge.bootstrap_ci(human, rater, judge.cohen_kappa)
    ci2, _ = judge.bootstrap_ci(human, rater, judge.cohen_kappa)
    assert ci1 == ci2 and und1 == 0
    assert ci1[0] <= judge.cohen_kappa(human, rater) <= ci1[1]
    a = judge.agreement(human, rater)
    assert a["n"] == 20 and a["percent_agreement"] == pytest.approx(0.6)
    assert a["kappa"]["ci95"] == ci1 and sum(map(sum, a["confusion_rows_human_cols_judge"])) == 20


# --------------------------------------------------------------------------- #
# Judge output and prompt
# --------------------------------------------------------------------------- #
def test_parse_judgement_and_derived_overall():
    raw = json.dumps({"claims": [
        {"claim": "Give 500 mg.", "label": "supported", "sources": ["[s1]"], "reason": "stated"},
        {"claim": "For 7 days.", "label": "unsupported", "sources": [], "reason": "not stated"}],
        "overall": "partial"})
    parsed, errors = judge.parse_judgement(raw, ["S1", "S2"])
    assert errors == [] and parsed["claims"][0]["sources"] == ["S1"]
    assert parsed["derived_overall"] == "partial" and parsed["consistent"]
    inconsistent = json.dumps({"claims": [{"claim": "x", "label": "contradicted", "sources": [], "reason": ""}],
                               "overall": "faithful"})
    p, _ = judge.parse_judgement(inconsistent, ["S1"])
    assert p["derived_overall"] == "no" and p["consistent"] is False
    assert judge.derived_overall(["supported", "supported"]) == "faithful"


@pytest.mark.parametrize(("raw", "message"), [
    ("nope", "not valid JSON"),
    (json.dumps({"claims": [], "overall": "faithful"}), "non-empty list"),
    (json.dumps({"claims": [{"claim": "x", "label": "maybe", "sources": [], "reason": ""}], "overall": "no"}),
     "label must be one of"),
    (json.dumps({"claims": [{"claim": "x", "label": "supported", "sources": ["S9"], "reason": ""}],
                 "overall": "faithful"}), "unknown sources"),
    (json.dumps({"claims": [{"claim": "x", "label": "supported", "sources": [], "reason": ""}], "overall": "yes"}),
     '"overall" must be one of'),
])
def test_parse_judgement_rejects(raw, message):
    parsed, errors = judge.parse_judgement(raw, ["S1"])
    assert parsed is None and any(message in e for e in errors), errors


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, format=None):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return type("R", (), {"content": reply, "stats": {"prompt_eval_count": 100, "eval_count": 50}})()


def test_judge_item_retries_once_and_hides_generator_quotes():
    item = {"item_id": "x-1", "question": "Q?", "answer": "Give 500 mg [S1].",
            "sources": [{"label": "S1", "text": "Give 500 mg every 8 hours."}], "faithful": ""}
    good = json.dumps({"claims": [{"claim": "Give 500 mg", "label": "supported", "sources": ["S1"], "reason": "r"}],
                       "overall": "faithful"})
    client = FakeClient([TimeoutError("timed out"), good])
    r = judge.judge_item(client, item)
    assert r["valid"] and r["overall"] == "faithful" and len(r["attempts"]) == 2
    prompt = client.calls[0][1]["content"]
    assert "[S1]\nGive 500 mg every 8 hours." in prompt and "Answer to check:\nGive 500 mg [S1]." in prompt
    assert "quote" not in prompt.lower()
    assert judge.OPTIONS["temperature"] == 0.0 and judge.JUDGE_MODEL == "qwen2.5:7b"


# --------------------------------------------------------------------------- #
# Sheet reading and the agree command
# --------------------------------------------------------------------------- #
def write_sheet(path: Path, labels: list[str]) -> None:
    header = ["item_id", "question", "answer", "source_1_label", "source_1_quote", "source_1_text",
              "source_2_label", "source_2_quote", "source_2_text", "faithful", "correct", "comment"]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i, lab in enumerate(labels):
            w.writerow([f"q-{i}", "Q", "A [S1]", "S1", "quote", "text one", "S1", "quote 2", "text one", lab, "", ""])


def test_read_sheet_dedupes_sources(tmp_path):
    write_sheet(tmp_path / "s.csv", ["Yes"])
    item = judge.read_sheet(tmp_path / "s.csv")[0]
    assert item["sources"] == [{"label": "S1", "text": "text one"}] and item["faithful"] == "yes"


def test_agree_command(tmp_path):
    write_sheet(tmp_path / "s.csv", ["yes", "partial", "no", ""])
    judged = [{"item_id": "q-0", "valid": True, "overall": "faithful", "derived_overall": "faithful"},
              {"item_id": "q-1", "valid": True, "overall": "no", "derived_overall": "partial"},
              {"item_id": "q-2", "valid": False, "overall": None, "derived_overall": None}]
    (tmp_path / "j.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged), encoding="utf-8")
    assert judge.main(["agree", "--sheet", str(tmp_path / "s.csv"), "--judgements", str(tmp_path / "j.jsonl")]) == 0
    out = json.loads((tmp_path / "agreement_j.json").read_text(encoding="utf-8"))
    assert out["n"] == 2 and out["percent_agreement"] == 0.5
    assert out["skipped"] == {"no human label": ["q-3"], "no valid judgement": ["q-2"]}
    assert out["disagreements"] == [{"item_id": "q-1", "human": "partial", "judge": "no"}]
    assert judge.main(["agree", "--sheet", str(tmp_path / "s.csv"), "--judgements", str(tmp_path / "j.jsonl"),
                       "--use", "derived"]) == 0
    assert json.loads((tmp_path / "agreement_j_derived.json").read_text(encoding="utf-8"))["percent_agreement"] == 1.0
    assert json.loads((tmp_path / "agreement_j.json").read_text(encoding="utf-8"))["percent_agreement"] == 0.5


def test_agree_rejects_unknown_labels(tmp_path):
    write_sheet(tmp_path / "s.csv", ["maybe"])
    (tmp_path / "j.jsonl").write_text("", encoding="utf-8")
    assert judge.main(["agree", "--sheet", str(tmp_path / "s.csv"), "--judgements", str(tmp_path / "j.jsonl")]) == 1


# --------------------------------------------------------------------------- #
# Labelling sheet
# --------------------------------------------------------------------------- #
def test_allocation_is_proportional():
    assert sheet.allocate({"cardiometabolic": 36, "aware": 16}, 30) == {"cardiometabolic": 21, "aware": 9}
    assert sum(sheet.allocate({"a": 5, "b": 5, "c": 5}, 7).values()) == 7


def test_committed_sheet_has_no_labels_or_automatic_checks():
    rows = list(csv.DictReader((ROOT / "notes" / "faithfulness_labels.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 30 and len({r["item_id"] for r in rows}) == 30
    cols = set(rows[0])
    assert {"faithful", "correct", "comment"} <= cols
    assert not any(k for k in cols if any(w in k for w in ("verified", "status", "route", "score", "gold")))
