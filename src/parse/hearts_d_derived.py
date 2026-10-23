"""Hand-transcribed, derived content for HEARTS-D flowcharts and dose statements.

Everything here was read off WHO's original PDF (``WHO-UCN-NCD-20.1-eng.pdf``, renders of
PDF pp. 16, 25 and 29) because the arrows of a flowchart are drawings, not text. It is
**derived**: it never replaces ``source_text`` and is never used to verify quotes.

``FLOW_TEXT``
    One explicit ``condition → action`` line per decision or step of each flowchart chunk,
    side branches included, following the arrows. Values (doses, thresholds, intervals) are
    copied exactly as printed, typos included ("80 g", "mI", "NaCI"); the retrieval
    ``text`` gets the documented typo corrections like every other body text. Where a box
    does not name the drug it changes ("increase dose to 1000 mg 1x daily"), the drug that
    the arrows lead from is added in brackets, e.g. "(metformin)". A printed value that
    conflicts with the rest of the document is followed by an editorial note in square
    brackets (``GLICLAZIDE_80_G_NOTE``).

``DOSE_STATEMENTS``
    One row per dose statement for ``data/interim/hearts_d_doses.csv``:

    * ``drug``: normalised lower-case name (e.g. ``metformin``, ``nph insulin``,
      ``glucose 50%``, ``sodium chloride 0.9%``); ``drug_as_printed``: the name as printed in
      the statement, empty when the statement names none; ``drug_source``: ``printed``, or
      ``inferred_from_sequence`` when the drug follows from the preceding flowchart boxes.
    * ``dose``: verbatim; ``dose_corrected``: ``dose`` with ``text_utils.TEXT_CORRECTIONS``
      applied (e.g. ``20–50 mI`` → ``20–50 ml``).
    * ``dose_basis``: ``per_dose``; ``daily_total_reference`` (a total daily dose mentioned to
      identify patients, not an instruction: Fig. 1 "2000 mg metformin and 160 mg gliclazide");
      ``titration_step`` (an increment or reduction); ``infusion_rate`` (a volume over a time).
    * ``dosing_as_printed``: the verbatim frequency/route/timing phrase, empty when none is
      printed, split into ``route`` (``oral`` or ``IV``, only when the statement says so, e.g.
      "IV", "intravenous", "ingest", "oral"; empty otherwise), ``frequency`` (e.g. "2 x daily",
      "at bedtime") and ``timing`` (e.g. "over 1–3 minutes", "morning", "in 2 hours").
    * ``step``: position along the branch and the action as printed; ``condition``: verbatim
      condition text, ``"; "``-separated where it comes from several boxes;
      ``discrepancies``: keys of ``who_hearts_d.DISCREPANCIES``.

    Every verbatim column is checked against the chunk that holds ``anchor`` by the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

# Editorial note after the printed "80 g" (Fig. 1); see the "gliclazide-unit" discrepancy.
GLICLAZIDE_80_G_NOTE = "[as printed; 80 mg elsewhere in this document, see source_discrepancy]"

# Keyed by chunk id without the "<doc_id>-" prefix.
FLOW_TEXT: dict[str, list[str]] = {
    "fig-1": [
        "If adult with symptoms of diabetes → test fasting or random plasma glucose (FPG or RPG)",
        "If asymptomatic adult who is 40+ years old and BMI ≥25 → test FPG",
        "If FPG ≥7 mmol/L and <18 mmol/L or RPG ≥11.1 mmol/L, and asymptomatic → repeat test on subsequent day",
        "If FPG ≥7 mmol/L and <18 mmol/L or RPG ≥11.1 mmol/L → counsel on diet and physical activity and "
        "adherence to medicines at all visits; review in 1 month (or immediately if symptoms appear)",
        "At the 1-month review, if goal not achieved → begin metformin 500 mg daily; "
        "review in 3 months (or immediately if symptoms appear)",
        "At the next 3-month review, if goal not achieved on metformin 500 mg daily → increase metformin to "
        "1000 mg 1 x daily; review in 3 months (or immediately if symptoms appear)",
        "At the next 3-month review, if goal not achieved on metformin 1000 mg 1 x daily → increase metformin to "
        "1000 mg 2 x daily; review in 3 months (or immediately if symptoms appear)",
        "At the next 3-month review, if goal not achieved on metformin 1000 mg 2 x daily → add gliclazide "
        "80 mg 1 x daily; review in 3 months (or immediately if symptoms appear)",
        "At the next 3-month review, if goal not achieved on gliclazide 80 mg 1 x daily → add gliclazide "
        "80 mg 2 x daily; review in 3 months (or immediately if symptoms appear)",
        "At the next 3-month review, if goal not achieved despite adherence to medication and diet and physical "
        "activity → refer to higher level of care or begin insulin (see insulin protocol)",
        "At the 1-month review or any of the next four 3-month reviews, if symptomatic and FPG ≥15 mmol/L → "
        "test urine ketones",
        "If FPG ≥15 mmol/L and symptomatic, or FPG/RPG >18 mmol/L regardless of symptoms → test urine ketones",
        "If urine ketones ≥2+ → refer to higher level of care",
        "If urine ketones <2+ → give metformin 1000 mg 2 x daily and gliclazide 80 g 2 x daily "
        f"{GLICLAZIDE_80_G_NOTE}; counsel on diet, physical activity and adherence to medication; review in 3–5 days",
        "At the 3–5 day review, if no improvement → refer to higher level of care",
        "At the 3–5 day review, if improvement → continue treatment; review in 2–3 months",
        "At the 2–3 month review, if goal not achieved → refer to higher level of care",
        "At the 2–3 month review, if goal achieved or frequent hypoglycaemia → consider reducing or stopping "
        "gliclazide in people who were on 2000 mg metformin and 160 mg gliclazide from diagnosis; "
        "review in 1 month",
    ],
    "fig-2-part1": [
        "If adult with symptoms of diabetes → test fasting or random plasma glucose (FPG or RPG)",
        "If adult 40+ years old with BMI >25 → test FPG",
        "If FPG ≥7 mmol/l and <18 mmol/l or RPG ≥11.1 mmol/l and <18 mmol/l # → counsel on diet and physical "
        "activity; review in 3 months",
        "At the 3-month review, if goal not achieved → begin metformin 500 mg once daily; counsel on diet and "
        "physical activity and adherence at all visits; review in 3 months",
        "At the next 3-month review, if goal not achieved → increase dose (metformin) to 1000 mg 1x daily; "
        "review in 3 months",
        "At the next 3-month review, if goal not achieved → increase dose (metformin) to 1000 mg 2x daily; "
        "review in 3 months",
        "At the next 3-month review, if goal not achieved → add gliclazide 80 mg 1x daily; counsel on "
        "hypoglycaemia at all subsequent visits; review in 3 months",
        "At the next 3-month review, if goal not achieved → increase dose (gliclazide) to 80 mg 2x daily; "
        "review in 3 months",
        "At the next 3-month review, if goal not achieved despite adherence to medication, healthy diet and "
        "physical activity → refer to higher-level health care facility for starting insulin*",
    ],
    "fig-2-part2": [
        "If FPG/RPG >18 mmol/l (325 mg/dl) → test urine ketones",
        "If ketones ≥2+ → refer to higher level of care",
        "If ketones <2+ → begin gliclazide 80 mg bid and counsel on diet modification, physical activity and "
        "adherence to medicines; review in 3–5 days",
        "At the 3–5 day review, if no improvement → refer to higher level of care",
        "At the 3–5 day review, if improvement → continue gliclazide and diet and physical activity; "
        "review in 2–3 months",
        "At the 2–3 month review, if goal not achieved → refer to higher level of care",
    ],
    "fig-2-part3": [
        "Screening for chronic complications, at every scheduled visit → measure blood pressure, review "
        "medication as per hypertension protocol",
        "Screening for chronic complications, upon diagnosis, and every two years thereafter, or as per "
        "ophthalmologist recommendation → refer for dilated-pupil retinal exam",
        "Screening for chronic complications, at every visit → examine feet for ulcers; if ulcer present → "
        "refer to higher level of care",
        "Screening for chronic complications, annually → assess risk of lower limb amputation (foot pulses, "
        "sensory neuropathy by monofilament, presence of healed or open ulcers, calluses); if ulcer present or "
        "pulse absent → refer to higher level of care",
        "Screening for chronic complications, annually → test for proteinuria; if positive → refer to higher "
        "level of care",
        "If severe hypoglycaemia (plasma glucose <50 mg/dl or 2.8 mmol/l) or signs, and conscious → give a "
        "sugar-sweetened drink",
        "If severe hypoglycaemia (plasma glucose <50 mg/dl or 2.8 mmol/l) or signs, and unconscious → give "
        "20–50 mI of 50% glucose (dextrose) IV over 1–3 minutes",
        "If severe hyperglycaemia (plasma glucose >18 mmol/l (325 mg/dl) and urine ketone 2+) or signs and "
        "symptoms of severe hyperglycaemia → set up intravenous drip 0.9% NaCI 1 litre in 2 hours; continue at "
        "1 litre every 4 hours; refer to hospital",
        "Goal for glycaemic control, fasting plasma glucose** → ≤7.0 mmol/l (126mg/dl)†",
    ],
    "annex1-flowchart": [
        "When starting insulin → start with 10 units intermediate-acting insulin (NPH) at bedtime; continue "
        "metformin and sulfonylurea; counsel patient on hypoglycaemia and on symptoms of hyperglycaemia; "
        "review in 3 days",
        "At the 3-day review, if FBG >7 mmol/L → increase insulin dosage by 1–2 units; review in 3 days",
        "After that review → continue increasing dosage by 1–2 units at 3-day intervals until FBG = 4–7 mmol/L",
        "If nocturnal hypoglycaemia occurs → do not increase insulin",
        "If FBG <4mmol/L or if nocturnal hypoglycaemia occurs → reduce insulin by 1–2 units",
        "If FBG 4–7 mmol/L and daytime hypoglycaemia occurs → reduce morning gliclazide by 40 mg",
        "If daytime hypoglycaemia continues after that → stop morning gliclazide",
        "If FBG >7mmol/L and nocturnal hypoglycaemia occurs → refer to higher level of care",
        "If stabilized (after titration, after reducing insulin, or after reducing or stopping gliclazide) → "
        "review with HbA1c in 3 months",
        "If HbA1c >7.5% → refer to higher level of care for intensification",
    ],
}


DRUG_SOURCES = ("printed", "inferred_from_sequence")
DOSE_BASES = ("per_dose", "daily_total_reference", "titration_step", "infusion_rate")
ROUTES = ("", "oral", "IV")


@dataclass(frozen=True)
class DoseStatement:
    anchor: str  # verbatim text that locates the chunk holding the statement
    drug: str  # normalised lower-case name
    drug_as_printed: str  # verbatim; empty when the statement names no drug
    dose: str  # verbatim (dose_corrected is derived from it with text_utils.TEXT_CORRECTIONS)
    dose_basis: str
    dosing_as_printed: str  # verbatim frequency/route/timing phrase, empty when none is printed
    route: str = ""
    frequency: str = ""
    timing: str = ""
    figure_or_section: str = ""
    step: str = ""
    branch: str = ""
    condition: str = ""
    discrepancies: tuple[str, ...] = ()

    @property
    def drug_source(self) -> str:
        return "printed" if self.drug_as_printed else "inferred_from_sequence"


_FIG1_MAIN = "main (left column)"
_FIG1_SIDE = "FPG ≥15 mmol/L and symptomatic, or FPG/RPG >18 mmol/L (right column)"
_FIG2_MAIN = "main (left column)"
_FIG2_SIDE = "FPG/RPG >18 mmol/l (middle column)"
_FIG2_ACUTE = "management of acute complications (right column)"
_REVIEW1 = "REVIEW IN 1 MONTH (or immediately if symptoms appear); IF goal not achieved"
_REVIEW3 = "REVIEW IN 3 MONTHS (or immediately if symptoms appear); IF goal not achieved"
_REVIEW3_FIG2 = "REVIEW IN 3 MONTHS; If goal not achieved"
_FIG1_SIDE_CONDITION = "FPG ≥15 mmol/L and SYMPTOMATIC or FPG/RPG >18 mmol/L, regardless of symptoms; Urine ketones <2+"
_FIG1_REDUCE = ("REVIEW in 2–3 months; IF goal achieved or frequent hypoglycaemia; "
                "in people who were on 2000 mg metformin and 160 mg gliclazide from diagnosis")
_SEVERE_HYPER = ("Severe hyperglycaemia (plasma glucose >18 mmol/l (325 mg/dl) and urine ketone 2+) or signs "
                 "and symptoms of severe hyperglycaemia")
_HYPO = "3 Prevention and management of complications of diabetes > Management of hypoglycaemia"
_HYPER = "3 Prevention and management of complications of diabetes > Management of hyperglycaemic emergencies"
_ANTIPLATELET = "3 Prevention and management of complications of diabetes > Antiplatelet treatment"

DOSE_STATEMENTS: list[DoseStatement] = [
    # Fig. 1, main branch
    DoseStatement("IF goal not achieved BEGIN METFORMIN 500 mg daily", "metformin", "METFORMIN", "500 mg",
                  "per_dose", "daily", frequency="daily", figure_or_section="Fig. 1", step="1 BEGIN",
                  branch=_FIG1_MAIN, condition=_REVIEW1, discrepancies=("first-review-interval",)),
    DoseStatement("INCREASE METFORMIN to 1000 mg 1 x daily", "metformin", "METFORMIN", "1000 mg", "per_dose",
                  "1 x daily", frequency="1 x daily", figure_or_section="Fig. 1", step="2 INCREASE",
                  branch=_FIG1_MAIN, condition=_REVIEW3),
    DoseStatement("INCREASE METFORMIN to 1000 mg 2 x daily", "metformin", "METFORMIN", "1000 mg", "per_dose",
                  "2 x daily", frequency="2 x daily", figure_or_section="Fig. 1", step="3 INCREASE",
                  branch=_FIG1_MAIN, condition=_REVIEW3),
    DoseStatement("ADD gliclazide 80 mg 1 x daily", "gliclazide", "gliclazide", "80 mg", "per_dose", "1 x daily",
                  frequency="1 x daily", figure_or_section="Fig. 1", step="4 ADD", branch=_FIG1_MAIN,
                  condition=_REVIEW3),
    DoseStatement("ADD gliclazide 80 mg 2 x daily", "gliclazide", "gliclazide", "80 mg", "per_dose", "2 x daily",
                  frequency="2 x daily", figure_or_section="Fig. 1", step="5 ADD", branch=_FIG1_MAIN,
                  condition=_REVIEW3),
    # Fig. 1, symptomatic / >18 mmol/L branch
    DoseStatement("GIVE METFORMIN 1000 mg 2 x daily", "metformin", "METFORMIN", "1000 mg", "per_dose", "2 x daily",
                  frequency="2 x daily", figure_or_section="Fig. 1", step="1 GIVE", branch=_FIG1_SIDE,
                  condition=_FIG1_SIDE_CONDITION, discrepancies=("hyperglycaemia-18", "hyperglycaemia-treatment")),
    DoseStatement("GLICLAZIDE 80 g 2 x daily", "gliclazide", "GLICLAZIDE", "80 g", "per_dose", "2 x daily",
                  frequency="2 x daily", figure_or_section="Fig. 1", step="1 GIVE", branch=_FIG1_SIDE,
                  condition=_FIG1_SIDE_CONDITION,
                  discrepancies=("gliclazide-unit", "hyperglycaemia-18", "hyperglycaemia-treatment")),
    DoseStatement("2000 mg metformin and 160 mg gliclazide from diagnosis", "metformin", "metformin", "2000 mg",
                  "daily_total_reference", "", figure_or_section="Fig. 1",
                  step="2 consider reducing or stopping gliclazide", branch=_FIG1_SIDE, condition=_FIG1_REDUCE),
    DoseStatement("2000 mg metformin and 160 mg gliclazide from diagnosis", "gliclazide", "gliclazide", "160 mg",
                  "daily_total_reference", "", figure_or_section="Fig. 1",
                  step="2 consider reducing or stopping gliclazide", branch=_FIG1_SIDE, condition=_FIG1_REDUCE),
    # Fig. 2, main branch
    DoseStatement("If goal not achieved BEGIN METFORMIN 500 mg once daily", "metformin", "METFORMIN", "500 mg",
                  "per_dose", "once daily", frequency="once daily", figure_or_section="Fig. 2", step="1 BEGIN",
                  branch=_FIG2_MAIN, condition=_REVIEW3_FIG2, discrepancies=("first-review-interval",)),
    DoseStatement("If goal not achieved increase dose to 1000 mg 1x daily", "metformin", "", "1000 mg", "per_dose",
                  "1x daily", frequency="1x daily", figure_or_section="Fig. 2", step="2 increase dose",
                  branch=_FIG2_MAIN, condition=_REVIEW3_FIG2),
    DoseStatement("If goal not achieved increase dose to 1000 mg 2x daily", "metformin", "", "1000 mg", "per_dose",
                  "2x daily", frequency="2x daily", figure_or_section="Fig. 2", step="3 increase dose",
                  branch=_FIG2_MAIN, condition=_REVIEW3_FIG2),
    DoseStatement("ADD gliclazide 80 mg 1x daily", "gliclazide", "gliclazide", "80 mg", "per_dose", "1x daily",
                  frequency="1x daily", figure_or_section="Fig. 2", step="4 ADD", branch=_FIG2_MAIN,
                  condition=_REVIEW3_FIG2),
    DoseStatement("If goal not achieved increase dose to 80 mg 2x daily", "gliclazide", "", "80 mg", "per_dose",
                  "2x daily", frequency="2x daily", figure_or_section="Fig. 2", step="5 increase dose",
                  branch=_FIG2_MAIN, condition=_REVIEW3_FIG2),
    # Fig. 2, >18 mmol/l branch
    DoseStatement("BEGIN gliclazide 80 mg bid", "gliclazide", "gliclazide", "80 mg", "per_dose", "bid",
                  frequency="bid", figure_or_section="Fig. 2", step="1 BEGIN", branch=_FIG2_SIDE,
                  condition="FPG/RPG >18 mmol/l ( 325 mg/dl); if ketones <2+",
                  discrepancies=("hyperglycaemia-18", "hyperglycaemia-treatment")),
    # Fig. 2, management of acute complications
    DoseStatement("give 20–50 mI of 50% glucose (dextrose) IV over 1–3 minutes", "glucose 50%",
                  "50% glucose (dextrose)", "20–50 mI", "per_dose", "IV over 1–3 minutes", route="IV",
                  timing="over 1–3 minutes", figure_or_section="Fig. 2", step="give", branch=_FIG2_ACUTE,
                  condition="Severe hypoglycaemia (plasma glucose <50 mg/dl or 2.8 mmol/l) or signs; If unconscious",
                  discrepancies=("severe-hypoglycaemia",)),
    DoseStatement("Set up intravenous drip 0.9% NaCI 1 litre in 2 hours", "sodium chloride 0.9%", "0.9% NaCI",
                  "1 litre", "infusion_rate", "in 2 hours", route="IV", timing="in 2 hours",
                  figure_or_section="Fig. 2", step="1 Set up intravenous drip", branch=_FIG2_ACUTE,
                  condition=_SEVERE_HYPER, discrepancies=("hyperglycaemia-18", "urine-ketones")),
    DoseStatement("continue at 1 litre every 4 hours", "sodium chloride 0.9%", "", "1 litre", "infusion_rate",
                  "every 4 hours", frequency="every 4 hours", figure_or_section="Fig. 2",
                  step="2 continue", branch=_FIG2_ACUTE, condition=_SEVERE_HYPER,
                  discrepancies=("hyperglycaemia-18", "urine-ketones")),
    # Annex 1
    DoseStatement("START with 10 units intermediate-acting insulin (NPH) at bedtime", "nph insulin",
                  "intermediate-acting insulin (NPH)", "10 units", "per_dose", "at bedtime", frequency="at bedtime",
                  figure_or_section="Annex 1", step="1 START", branch="main"),
    DoseStatement("INCREASE INSULIN dosage by 1–2 units", "insulin", "INSULIN", "1–2 units", "titration_step", "",
                  figure_or_section="Annex 1", step="2 INCREASE INSULIN dosage by", branch="main",
                  condition="REVIEW in 3 days; IF FBG >7 mmol/L"),
    DoseStatement("CONTINUE INCREASING DOSAGE by 1–2 units at 3-day intervals", "insulin", "", "1–2 units",
                  "titration_step", "at 3-day intervals", frequency="at 3-day intervals", figure_or_section="Annex 1",
                  step="3 CONTINUE INCREASING DOSAGE by", branch="main",
                  condition="REVIEW in 3 days; until FBG = 4–7 mmol/L; Do not increase insulin if nocturnal "
                            "hypoglycaemia occurs"),
    DoseStatement("REDUCE INSULIN by 1–2 units", "insulin", "INSULIN", "1–2 units", "titration_step", "",
                  figure_or_section="Annex 1", step="4 REDUCE INSULIN by", branch="main",
                  condition="IF FBG <4mmol/L or if nocturnal hypoglycaemia occurs"),
    DoseStatement("REDUCE morning gliclazide by 40 mg", "gliclazide", "gliclazide", "40 mg", "titration_step",
                  "morning", timing="morning", figure_or_section="Annex 1", step="4 REDUCE morning gliclazide by",
                  branch="main", condition="IF FBG 4–7 mmol/L and daytime hypoglycaemia occurs"),
    # Prose
    DoseStatement("they should ingest 15–20 g of glucose", "glucose", "glucose", "15–20 g", "per_dose", "",
                  route="oral", figure_or_section=_HYPO, step="ingest",
                  condition="If the patient is able to ingest food or drink"),
    DoseStatement("give oral carbohydrate that contains 15–20 g of rapidly absorbing forms of glucose",
                  "carbohydrate", "oral carbohydrate", "15–20 g", "per_dose", "", route="oral",
                  figure_or_section=_HYPO, step="give", condition="If glucose is not available"),
    DoseStatement("(20–50 mL of 50% glucose over 1–3 minutes)", "glucose 50%", "50% glucose", "20–50 mL",
                  "per_dose", "over 1–3 minutes", route="IV",
                  timing="over 1–3 minutes", figure_or_section=_HYPO,
                  step="given hypertonic glucose (dextrose) intravenously",
                  condition="Unconscious patients, those with plasma glucose ≤2.8 mmol/L (50 mg/dL) and those "
                            "unable to to ingest drink", discrepancies=("severe-hypoglycaemia",)),
    DoseStatement("Infuse isotonic saline (0.9% NaCl) at a rate of 1000 mL in the first 2 hours",
                  "sodium chloride 0.9%", "isotonic saline (0.9% NaCl)", "1000 mL", "infusion_rate",
                  "in the first 2 hours", timing="in the first 2 hours", figure_or_section=_HYPER,
                  step="Infuse", condition="suspected DKA or HHS"),
    DoseStatement("Continue with 1000 mL every 4 hours until reaching a hospital", "sodium chloride 0.9%", "",
                  "1000 mL", "infusion_rate", "every 4 hours until reaching a hospital", frequency="every 4 hours",
                  timing="until reaching a hospital", figure_or_section=_HYPER, step="Continue",
                  condition="suspected DKA or HHS"),
    DoseStatement("75–100 mg of acetylsalicylic acid daily", "acetylsalicylic acid", "acetylsalicylic acid",
                  "75–100 mg", "per_dose", "daily", frequency="daily", figure_or_section=_ANTIPLATELET,
                  step="prescribed",
                  condition="all people with diabetes who have survived a CVD event and have no history of major "
                            "bleeding"),
]
