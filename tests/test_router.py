"""Router (generation v2): personal-advice detector, drug lexicon and grounding, weight-based dose lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluate.relevance import load_chunks
from src.generate.router import (
    EXTRA_DRUGS,
    band_contains,
    build_lexicon,
    detect_personal_advice,
    lookup_doses,
    parse_weight_band,
    parse_weight_kg,
    ungrounded_drugs,
)

ROOT = Path(__file__).resolve().parents[1]
PERSONAL = {"pa-001", "pa-002", "pa-003", "pa-004"}


@pytest.fixture(scope="module")
def chunks() -> dict[str, dict]:
    return {c["chunk_id"]: c for c in load_chunks()}


@pytest.fixture(scope="module")
def lexicon(chunks):
    return build_lexicon(list(chunks.values()))


def questions() -> list[dict]:
    path = ROOT / "eval" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bge_top5() -> dict[str, list[str]]:
    run = json.loads((ROOT / "results" / "retrieval" / "bge-base.json").read_text(encoding="utf-8"))
    return {i["qid"]: [t["chunk_id"] for t in i["top"][:5]] for i in run["items"]}


# --------------------------------------------------------------------------- #
# a. Personal advice
# --------------------------------------------------------------------------- #
def test_personal_advice_items_are_detected(lexicon):
    qs = {q["qid"]: q["question"] for q in questions()}
    for qid in sorted(PERSONAL):
        assert detect_personal_advice(qs[qid], lexicon), qid


def test_no_other_eval_item_is_detected(lexicon):
    """Includes na-001 ("Does that mean I need antibiotics?") and the clinician questions
    "My patient ..." (htn-002) and "My diabetic patient's ... Should I treat it" (cmb-002)."""
    wrong = [q["qid"] for q in questions() if q["qid"] not in PERSONAL and detect_personal_advice(q["question"], lexicon)]
    assert wrong == []
    assert not detect_personal_advice(next(q["question"] for q in questions() if q["qid"] == "na-001"), lexicon)


@pytest.mark.parametrize(("question", "expected"), [
    ("Can I skip my evening insulin if I'm not hungry?", True),
    ("Should we give our son the rest of his antibiotics?", True),
    ("I want to stop taking lisinopril. Is that OK?", True),
    ("Can I increase the dose of my tablets?", True),
    ("Should I double the dose?", True),             # "dose" is a medicine word
    ("Can I take a walk after dinner?", False),      # no medicine word
    ("Should a patient stop metformin before surgery?", False),
    ("What dose of amoxicillin should a child take?", False),  # no first person
])
def test_personal_advice_rules(lexicon, question, expected):
    assert detect_personal_advice(question, lexicon) is expected


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #
def test_lexicon_sources(lexicon):
    names = set(lexicon.aliases)
    assert {"amoxicillin", "cefalexin", "sulfamethoxazole + trimethoprim", "amoxicillin + clavulanic acid"} <= names
    assert {"metformin", "gliclazide", "insulin", "nph insulin", "acetylsalicylic acid"} <= names  # HEARTS-D CSV
    assert {"amlodipine", "telmisartan", "lisinopril", "glibenclamide"} <= names  # found in the chunks
    assert "losartan" not in names  # a candidate that occurs in no chunk
    assert set(EXTRA_DRUGS) <= names
    assert not {"glucose", "carbohydrate", "co-trimoxazole", "aspirin"} & names  # excluded / aliases only


@pytest.mark.parametrize(("text", "expected"), [
    ("A child on co-trimoxazole", ["sulfamethoxazole + trimethoprim"]),
    ("Sulfamethoxazole+ trimethoprim (oral)", ["sulfamethoxazole + trimethoprim"]),
    ("an 8 kg baby; what dose of trimethoprim", ["trimethoprim"]),  # not the combination
    ("Amoxicillin+clavulanic acid or AMOXICILLIN", ["amoxicillin + clavulanic acid", "amoxicillin"]),
    ("daily aspirin", ["acetylsalicylic acid"]),
    ("start intermediate-acting insulin (NPH) or insulin", ["nph insulin", "insulin"]),
    ("Empagliflozin 10 mg", ["empagliflozin"]),
    ("amoxicillinx, metforminic", []),  # whole words only
    ("random glucose of 20 mmol/L", []),
])
def test_lexicon_matching(lexicon, text, expected):
    assert lexicon.find(text) == expected


def test_lexicon_families_and_mentions(lexicon):
    assert lexicon.family("nph insulin") == "insulin"
    assert lexicon.mentioned_in("insulin", "start NPH insulin at bedtime")
    assert lexicon.mentioned_in("acetylsalicylic acid", "give aspirin 75 mg")


# --------------------------------------------------------------------------- #
# b. Drug grounding
# --------------------------------------------------------------------------- #
def test_drug_grounding_on_the_eval_set(chunks, lexicon):
    """With the stored bge-base top 5, only nm-005 (empagliflozin) and pa-003 (amlodipine) name a
    drug that no retrieved chunk mentions; every answerable item passes. pa-003 never reaches this
    check: the personal-advice route comes first."""
    top = bge_top5()
    refused = {q["qid"]: ungrounded_drugs(q["question"], [chunks[c] for c in top[q["qid"]]], lexicon)
               for q in questions()}
    assert {k: v for k, v in refused.items() if v} == {"nm-005": ["empagliflozin"], "pa-003": ["amlodipine"]}
    assert detect_personal_advice(next(q["question"] for q in questions() if q["qid"] == "pa-003"), lexicon)


def test_drug_grounding_unit(chunks, lexicon):
    fig1 = chunks["who-hearts-d-2020-fig-1"]
    assert ungrounded_drugs("Can metformin be used?", [fig1], lexicon) == []
    assert ungrounded_drugs("What dose of sitagliptin?", [fig1], lexicon) == ["sitagliptin"]
    assert ungrounded_drugs("How should hypertension be treated?", [], lexicon) == []  # no drug named


# --------------------------------------------------------------------------- #
# c. Weight-based dose lookup
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("text", "kg"), [
    ("a 12 kg child", 12.0), ("weighing 35kg", 35.0), ("an 8-kg baby", 8.0), ("25 mg/kg", None), ("eGFR 25", None),
])
def test_parse_weight(text, kg):
    assert parse_weight_kg(text) == kg


@pytest.mark.parametrize(("band", "parsed"), [
    ("10–< 15 kg", (10, 15, True)), ("3–6 kg", (3, 6, True)), ("≥ 20kg", (20, float("inf"), True)),
    ("> 45 kg (> 12 years)", (45, float("inf"), False)), ("< 45 kg (< 12 years)", (0, 45, True)), ("", None),
])
def test_parse_weight_band(band, parsed):
    assert parse_weight_band(band) == parsed


def test_band_edges():
    assert band_contains("10–< 15 kg", 10) and not band_contains("10–< 15 kg", 15)
    assert band_contains("≥ 30 kg", 30) and not band_contains("> 45 kg (> 12 years)", 45)


@pytest.mark.parametrize(("qid", "drug", "dose", "frequency", "chunk"), [
    ("aw-005", "Amoxicillin", "500 mg", "every 12 hours", "who-aware-2022-ch12-dosing-t12-5-children"),
    ("dl-001", "Cefalexin", "375 mg", "every 12 hours", "who-aware-2022-ch16-dosing-children"),
    ("dl-005", "Trimethoprim", "40 mg", "every 12 hours", "who-aware-2022-ch23-dosing-children"),
    ("pv-001", "Sulfamethoxazole+ trimethoprim", "800 mg + 160 mg", "every 12 hours",
     "who-aware-2022-ch23-dosing-children"),
])
def test_dose_lookup_matches_the_gold_rows(lexicon, qid, drug, dose, frequency, chunk):
    q = next(q for q in questions() if q["qid"] == qid)
    lookup = lookup_doses(q["question"], lexicon)
    assert lookup is not None and len(lookup.rows) == 1
    row = lookup.rows[0]
    assert (row["antibiotic"], row["resolved_dose"], row["resolved_frequency"], row["chunk_id"]) == (
        drug, dose, frequency, chunk)
    gold = q["gold_structured"]["any_of"][0]
    assert gold["dose"] == row["resolved_dose"] and gold["frequency"] == row["resolved_frequency"]


def test_dose_lookup_adds_the_adult_chunk_for_adult_dose_rows(lexicon):
    lookup = lookup_doses("A child weighing 35 kg has a lower urinary tract infection. What dose of co-trimoxazole?",
                          lexicon)
    assert lookup.chunk_ids() == ["who-aware-2022-ch23-dosing-children", "who-aware-2022-ch23-dosing-adults"]


def test_dose_lookup_lists_all_drugs_when_none_is_named(lexicon):
    lookup = lookup_doses("What should an 8 kg baby with a urine infection be given?", lexicon)
    drugs = {r["antibiotic"] for r in lookup.rows}
    assert {"Trimethoprim", "Sulfamethoxazole+ trimethoprim"} <= drugs and len(drugs) >= 3
    assert all(band_contains(r["weight_band"], 8) for r in lookup.rows)


def test_dose_lookup_refuses_or_skips(lexicon):
    no_row = lookup_doses("A 2 kg newborn has a urine infection. What dose of trimethoprim?", lexicon)
    assert no_row is not None and no_row.rows == ()  # below every band -> refuse
    wrong_drug = lookup_doses("A 12 kg child has impetigo. What dose of nitrofurantoin?", lexicon)
    assert wrong_drug is not None and wrong_drug.rows == ()
    assert lookup_doses("What dose of amoxicillin for a 12 kg child?", lexicon) is None  # no infection
    assert lookup_doses("What dose of amoxicillin for pneumonia?", lexicon) is None  # no weight
    for q in questions():  # only the four weight-based items take this route
        if q["qid"] not in {"aw-005", "dl-001", "dl-005", "pv-001"}:
            assert lookup_doses(q["question"], lexicon) is None, q["qid"]
