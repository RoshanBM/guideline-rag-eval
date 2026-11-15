"""Routing before the refusal gate (generation pipeline v2).

``Answerer.answer`` (``src/generate/answer.py``) applies these checks in order; the first that
fires decides the route:

a. **Personal advice** (:func:`detect_personal_advice`): a first-person (or own-family) medication
   action, e.g. "Should I take ...", "Can I stop taking my ...", "Can I give her ... leftover
   antibiotic". The answer is the fixed :data:`~src.generate.config.REDIRECT_MESSAGE`; the LLM is
   not called. The rules are deliberately narrow: they need (1) "I"/"we" with a modal and one of
   the action verbs (take, stop, double, increase, reduce, give, skip, use, ...) or an action verb
   followed by "my/our/her/his"; (2) a medicine word (a lexicon drug, "antibiotic", "tablet",
   "medicine", ...) or "leftover"; and (3) no "patient" in the question (clinician questions such
   as "My patient ... should I treat ..." are not personal advice).
b. **Drug grounding** (:func:`ungrounded_drugs`): every drug the question names (lexicon match)
   must appear in at least one of the top-5 retrieved chunks; otherwise the question is refused
   (it asks about a drug the retrieved guideline text does not mention, e.g. empagliflozin).
c. **Weight-based dose lookup** (:func:`lookup_doses`): if the question gives a weight in kg and
   names an infection that maps to ``data/interim/aware_dosing.csv``, the rows for that infection
   (and the named drug, if any) whose weight band contains the weight are selected. Only those rows
   and their chunks go to the LLM, which phrases the answer. If no row matches, the question is
   refused. A question with a weight but no recognised infection is not routed here.
d. Otherwise: the refusal gate, then the LLM.

**Drug lexicon** (:func:`build_lexicon`): the antibiotics in ``aware_dosing.csv``, the drugs in
``hearts_d_doses.csv`` (without glucose, oral carbohydrate and saline, which also occur as lab
values and fluids), the names in :data:`CORPUS_DRUG_CANDIDATES` that occur in the hypertension or
HEARTS-D chunks, and :data:`EXTRA_DRUGS` (common drugs absent from the corpus, so that questions
about them can be recognised). Each entry has a canonical name, aliases (``co-trimoxazole`` for
sulfamethoxazole + trimethoprim, ``aspirin`` for acetylsalicylic acid) and a family (NPH insulin is
in the insulin family). Matching is case-insensitive, on whole words, longest alias first, with
"+" and hyphen spacing folded.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.parse.schema import verifiable_text
from src.parse.text_utils import normalise_for_match

ROOT = Path(__file__).resolve().parents[2]
AWARE_DOSING = ROOT / "data" / "interim" / "aware_dosing.csv"
HEARTS_DOSES = ROOT / "data" / "interim" / "hearts_d_doses.csv"

# --------------------------------------------------------------------------- #
# Drug lexicon
# --------------------------------------------------------------------------- #
CORPUS_DRUG_CANDIDATES = (
    # antihypertensives
    "amlodipine", "atenolol", "bisoprolol", "candesartan", "captopril", "carvedilol", "chlorthalidone",
    "chlortalidone", "diltiazem", "doxazosin", "enalapril", "felodipine", "furosemide", "hydralazine",
    "hydrochlorothiazide", "indapamide", "irbesartan", "labetalol", "lisinopril", "losartan", "methyldopa",
    "metoprolol", "nifedipine", "olmesartan", "perindopril", "propranolol", "ramipril", "spironolactone",
    "telmisartan", "valsartan", "verapamil",
    # diabetes and cardiovascular
    "glibenclamide", "glimepiride", "glipizide", "atorvastatin", "lovastatin", "pravastatin", "rosuvastatin",
    "simvastatin", "clopidogrel",
)
EXTRA_DRUGS = (
    "empagliflozin", "dapagliflozin", "canagliflozin", "sitagliptin", "linagliptin", "pioglitazone",
    "liraglutide", "semaglutide", "clindamycin", "doxycycline", "ibuprofen", "paracetamol", "naproxen",
    "prednisolone", "warfarin", "levofloxacin", "moxifloxacin", "cefuroxime", "flucloxacillin", "omeprazole",
)
ALIASES: dict[str, tuple[str, ...]] = {
    "sulfamethoxazole + trimethoprim": ("co-trimoxazole", "cotrimoxazole", "trimethoprim-sulfamethoxazole",
                                        "sulfamethoxazole/trimethoprim"),
    "amoxicillin + clavulanic acid": ("co-amoxiclav", "amoxicillin-clavulanate", "amoxicillin/clavulanic acid"),
    "acetylsalicylic acid": ("aspirin",),
    "nph insulin": ("intermediate-acting insulin", "isophane insulin"),
    "glibenclamide": ("glyburide",),
    "chlorthalidone": ("chlortalidone",),
    "paracetamol": ("acetaminophen",),
}
FAMILY = {"nph insulin": "insulin", "chlortalidone": "chlorthalidone"}
HEARTS_EXCLUDED = {"glucose", "glucose 50%", "carbohydrate", "sodium chloride 0.9%"}


def canonical_drug(name: str) -> str:
    t = normalise_for_match(name).lower()
    t = re.sub(r"\s*\+\s*", " + ", t)
    return re.sub(r"\s+", " ", t).strip()


def _alias_pattern(alias: str) -> str:
    parts = []
    for token in re.split(r"(\s*\+\s*|\s+|-|/)", alias):
        if not token:
            continue
        if token.strip() == "+":
            parts.append(r"\s*\+\s*")
        elif token.isspace() or token in ("-", "/"):
            parts.append(r"[\s/-]*")
        else:
            parts.append(re.escape(token))
    return r"(?<![a-z])" + "".join(parts) + r"(?![a-z])"


@dataclass
class DrugLexicon:
    aliases: dict[str, tuple[str, ...]]  # canonical -> all names, canonical first
    _patterns: list[tuple[re.Pattern[str], str]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        pairs = [(a, c) for c, names in self.aliases.items() for a in names]
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
        self._patterns = [(re.compile(_alias_pattern(a)), c) for a, c in pairs]

    @staticmethod
    def _prep(text: str) -> str:
        return normalise_for_match(text).lower()

    def find(self, text: str) -> list[str]:
        """Canonical drugs named in ``text``, in order of first mention (longest alias wins)."""
        t = self._prep(text)
        taken = [False] * len(t)
        hits: list[tuple[int, str]] = []
        for pat, canon in self._patterns:
            for m in pat.finditer(t):
                if any(taken[m.start():m.end()]):
                    continue
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
                hits.append((m.start(), canon))
        out: list[str] = []
        for _, canon in sorted(hits):
            if canon not in out:
                out.append(canon)
        return out

    def mentioned_in(self, canon: str, text: str) -> bool:
        """True if any name of ``canon`` occurs in ``text`` (anywhere, even inside a longer name)."""
        t = self._prep(text)
        return any(re.search(_alias_pattern(a), t) for a in self.aliases[canon])

    @staticmethod
    def family(canon: str) -> str:
        return FAMILY.get(canon, canon)


def chunk_drug_text(record: dict[str, Any]) -> str:
    """Text searched for drug names: source text, footnotes and the flowchart reading guide."""
    return " ".join(t for t in (verifiable_text(record), record.get("flow_text") or "") if t)


def build_lexicon(chunks: list[dict[str, Any]]) -> DrugLexicon:
    names: set[str] = set()
    with AWARE_DOSING.open(encoding="utf-8") as fh:
        names |= {canonical_drug(r["antibiotic"]) for r in csv.DictReader(fh) if r["antibiotic"].strip()}
    with HEARTS_DOSES.open(encoding="utf-8") as fh:
        names |= {canonical_drug(r["drug"]) for r in csv.DictReader(fh) if r["drug"].strip()} - HEARTS_EXCLUDED
    corpus = " ".join(chunk_drug_text(c) for c in chunks
                      if c.get("doc_id") in ("who-htn-2021", "who-hearts-d-2020")).lower()
    names |= {d for d in CORPUS_DRUG_CANDIDATES if re.search(rf"(?<![a-z]){d}(?![a-z])", corpus)}
    names |= set(EXTRA_DRUGS)
    names -= {a for aliases in ALIASES.values() for a in aliases}  # an alias is not its own entry
    return DrugLexicon({n: (n, *ALIASES.get(n, ())) for n in sorted(names)})


@lru_cache(maxsize=1)
def default_lexicon() -> DrugLexicon:
    from src.evaluate.relevance import load_chunks

    return build_lexicon(load_chunks())


# --------------------------------------------------------------------------- #
# a. Personal advice
# --------------------------------------------------------------------------- #
_ACTIONS = r"(?:take|taking|stop|double|increase|reduce|give|skip|use|halve|cut down|restart|share)"
_PERSONAL_ACTION_RES = (
    re.compile(rf"\b(?:should|can|could|may|shall|must|do|would)\s+(?:i|we)\s+(?:\w+\s+)?{_ACTIONS}\b"),
    re.compile(rf"\b(?:i|we)\s+(?:want|need|plan|am going|are going|have been told)\s+to\s+{_ACTIONS}\b"),
    re.compile(rf"\b{_ACTIONS}\s+(?:taking\s+)?(?:my|our|her|his)\b"),
)
_MEDICINE_WORDS = re.compile(
    r"\b(?:antibiotics?|medicines?|medications?|tablets?|pills?|capsules?|doses?|drugs?|insulin|inhalers?|"
    r"syrup|left\s?over|left over)\b")


def detect_personal_advice(question: str, lexicon: DrugLexicon | None = None) -> bool:
    q = normalise_for_match(question).lower()
    if re.search(r"\bpatients?\b", q):
        return False
    if not any(p.search(q) for p in _PERSONAL_ACTION_RES):
        return False
    lexicon = lexicon or default_lexicon()
    return bool(_MEDICINE_WORDS.search(q) or lexicon.find(question))


# --------------------------------------------------------------------------- #
# b. Drug grounding
# --------------------------------------------------------------------------- #
def ungrounded_drugs(question: str, records: list[dict[str, Any]], lexicon: DrugLexicon | None = None) -> list[str]:
    """Drugs named in ``question`` that appear in none of ``records`` (the top-5 chunks)."""
    lexicon = lexicon or default_lexicon()
    text = " ".join(chunk_drug_text(r) for r in records)
    return [d for d in lexicon.find(question) if not lexicon.mentioned_in(d, text)]


# --------------------------------------------------------------------------- #
# c. Weight-based dose lookup
# --------------------------------------------------------------------------- #
WEIGHT_RE = re.compile(r"(?<![\w/.])(\d+(?:\.\d+)?)\s*-?\s*kg\b")
INFECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"pneumonia", "Community-acquired pneumonia – mild"),
    (r"impetigo|cellulitis|erysipelas|skin (?:and soft tissue )?infection",
     "Skin and soft tissue infections – mild bacterial impetigo, erysipelas and cellulitis"),
    (r"urin(?:e|ary)(?: tract)? infection|\butis?\b|bladder infection|cystitis", "Lower urinary tract infection"),
    (r"\bear infection|otitis", "Acute otitis media"),
    (r"sore throat|strep throat|pharyngitis|tonsillitis", "Pharyngitis"),
    (r"sinusitis|sinus infection", "Acute sinusitis"),
    (r"diarrhoea|diarrhea|gastroenteritis", "Acute infectious diarrhoea/gastroenteritis"),
    (r"dental|tooth|teeth", "Oral and dental infections"),
    (r"eye infection|conjunctivitis", "Bacterial eye infections (excluding trachoma)"),
    (r"lymphadenitis", "Localized acute bacterial lymphadenitis"),
    (r"typhoid|enteric fever", "Enteric fever"),
    (r"\bbites?\b|wound infection", "Wound and bite-related infections"),
    (r"\bburns?\b", "Burn wound-related infections"),
    (r"trachoma", "Trachoma"),
    (r"\bcopd\b|chronic obstructive", "Exacerbation of chronic obstructive pulmonary disease"),
    (r"chlamydia", "Sexually transmitted infections – chlamydial urogenital infection"),
    (r"gonorrh", "Sexually transmitted infections – gonococcal infection"),
    (r"syphilis", "Sexually transmitted infections – syphilis"),
    (r"trichomon", "Sexually transmitted infections – trichomoniasis"),
)


def parse_weight_kg(question: str) -> float | None:
    m = WEIGHT_RE.search(normalise_for_match(question).lower())
    return float(m.group(1)) if m else None


def match_infection(question: str) -> str | None:
    q = normalise_for_match(question).lower()
    return next((inf for pat, inf in INFECTION_PATTERNS if re.search(pat, q)), None)


def parse_weight_band(band: str) -> tuple[float, float, bool] | None:
    """``(low, high, low_inclusive)`` for a printed band; the high end is exclusive.

    "10–< 15 kg" -> (10, 15, True); "3–6 kg" -> (3, 6, True) (read as 3–< 6, like the other bands);
    "≥ 20 kg" -> (20, inf, True); "> 45 kg (> 12 years)" -> (45, inf, False); "< 45 kg" -> (0, 45, True).
    """
    b = normalise_for_match(band).lower().replace(" ", "")
    if m := re.match(r"(\d+(?:\.\d+)?)-<?(\d+(?:\.\d+)?)kg", b):
        return float(m.group(1)), float(m.group(2)), True
    if m := re.match(r"(≥|>=)(\d+(?:\.\d+)?)kg", b):
        return float(m.group(2)), float("inf"), True
    if m := re.match(r">(\d+(?:\.\d+)?)kg", b):
        return float(m.group(1)), float("inf"), False
    if m := re.match(r"<(\d+(?:\.\d+)?)kg", b):
        return 0.0, float(m.group(1)), True
    return None


def band_contains(band: str, weight: float) -> bool:
    parsed = parse_weight_band(band)
    if parsed is None:
        return False
    lo, hi, lo_incl = parsed
    return (weight >= lo if lo_incl else weight > lo) and weight < hi


@lru_cache(maxsize=1)
def dosing_rows() -> tuple[dict[str, str], ...]:
    with AWARE_DOSING.open(encoding="utf-8") as fh:
        return tuple(csv.DictReader(fh))


@dataclass(frozen=True)
class DoseLookup:
    weight_kg: float
    infection: str
    drugs: tuple[str, ...]  # canonical drugs named in the question (empty: every drug for the infection)
    rows: tuple[dict[str, str], ...]  # matching rows; empty -> refuse

    def chunk_ids(self) -> list[str]:
        """Chunks of the rows, plus the adult-table chunk a row's dose was resolved from."""
        out: list[str] = []
        for r in self.rows:
            # resolved_from is "<chunk_id> | <antibiotic> | <route>"
            for cid in (r["chunk_id"], (r.get("resolved_from") or "").split("|")[0].strip()):
                if cid and cid not in out:
                    out.append(cid)
        return out


def lookup_doses(question: str, lexicon: DrugLexicon | None = None,
                 rows: tuple[dict[str, str], ...] | None = None) -> DoseLookup | None:
    """The weight-band rows for the question, or None if it is not a weight-based dose question."""
    weight = parse_weight_kg(question)
    infection = match_infection(question)
    if weight is None or infection is None:
        return None
    lexicon = lexicon or default_lexicon()
    rows = dosing_rows() if rows is None else rows
    drugs = tuple(lexicon.find(question))
    matched = tuple(r for r in rows if r["infection"] == infection and r["weight_band"]
                    and band_contains(r["weight_band"], weight)
                    and (not drugs or canonical_drug(r["antibiotic"]) in drugs))
    return DoseLookup(weight, infection, drugs, matched)


def dose_row_text(row: dict[str, str], label: str) -> str:
    parts = [f"{row['antibiotic']} ({row['route'] or 'route not stated'})",
             f"population: {row['population'] or 'not stated'}, weight band {row['weight_band']}",
             f"dose: {row['resolved_dose']}" + (f" {row['resolved_frequency']}" if row["resolved_frequency"] else ""),
             f"duration: {row['duration'] or 'not stated'}",
             f"line: {row['line'].replace('_', ' ')}"]
    if row.get("resolved_rule") == "adult_dose_lookup":
        parts.append("the table says to use the adult dose at this weight")
    for key in ("dose_unit_note", "notes"):
        if row.get(key):
            parts.append(f"note: {row[key]}")
    return f"[{label}] " + " | ".join(parts)
