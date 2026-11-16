"""Check a generated answer against its sources.

* :func:`verify_citations`: each citation's label is mapped to its source chunk and the quote
  is checked with ``text_utils.quote_in_text`` against the chunk's ``source_text`` plus
  footnotes only (``schema.verifiable_text``); ``context_header``, ``flow_text`` and
  ``aware_category_note`` never count. Matching folds whitespace, quote and dash styles and the
  documented typo corrections, and keeps case. Edits at the quote's edges that do not change
  what the quoted words say are tolerated (:func:`quote_variants`):

  1. surrounding quote marks;
  2. a trailing ellipsis ("..." or "…"), then one trailing full stop, so a quote that stops
     mid-sentence with "." or "..." still matches;
  3. a different case of the **first letter only** ("even though" / "Even though"), from
     starting a quote mid-sentence. Case elsewhere must match ("GIVE METFORMIN" is not
     "give metformin").

  Anything else (an ellipsis inside the quote, added or dropped words or punctuation, a label
  inside the quote) must match verbatim. Each check has a ``status``: ``verified``;
  ``verified_wrong_label`` (the quote does not match its own source but matches another source
  shown to the model, under the same tolerance; ``correct_label`` names it); or
  ``not_verified``. ``tolerant`` is true when the match needed rule 2 or 3. ``verified`` is
  true only for ``verified``.
* :func:`unlabelled_number_sentences`: sentences of the answer that contain a number but no
  ``[Sn]`` label (labels that open the next sentence, as in "... once a day. [S1]", count for
  the sentence before).
* :func:`unsupported_numbers`: every number + unit in the answer (``500 mg``, ``3-5 days``,
  ``80 g``, ``140 mmHg``, ``50%``) must appear in the source text of at least one **cited**
  chunk. A range is supported if the same range, or both of its endpoints, appear; a number in
  a source range (``3-5 days``) also supports its endpoints (``5 days``). Values that also occur
  in the question (e.g. "a 12 kg child") are reported with ``in_question: true`` and are not
  counted as unsupported by the metrics.
"""

from __future__ import annotations

import re
from typing import Any

from src.parse.schema import verifiable_text
from src.parse.text_utils import normalise_for_match, quote_in_text

_QUOTE_MARKS = "\"'“”‘’"

_NUM = r"\d+(?:\.\d+)?"
_UNIT_CANON = {
    "mg/kg/day": "mg/kg/day", "mg/kg": "mg/kg", "mmol/mol": "mmol/mol", "mmol/l": "mmol/l", "mg/dl": "mg/dl",
    "mmhg": "mmhg", "mcg": "mcg", "µg": "mcg", "μg": "mcg", "mg": "mg", "g": "g", "kg": "kg", "ml": "ml",
    "iu": "iu", "units": "unit", "unit": "unit", "%": "%",
    "days": "day", "day": "day", "weeks": "week", "week": "week", "hours": "hour", "hour": "hour",
    "minutes": "minute", "minute": "minute", "min": "minute", "months": "month", "month": "month",
    "years": "year", "year": "year",
}
_UNIT_RE = "|".join(re.escape(u) for u in sorted(_UNIT_CANON, key=len, reverse=True))
NUMBER_UNIT_RE = re.compile(
    rf"(?<![\w.])(?P<lo>{_NUM})(?:\s*(?:-|to)\s*(?P<hi>{_NUM}))?\s*(?P<unit>{_UNIT_RE})(?![a-zµμ])")


def _prepare(text: str) -> str:
    text = normalise_for_match(text).lower()
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)  # thousands separators: 1,000 -> 1000
    # space as thousands separator, as AWaRe prints "800 000 IU" -> 800000
    return re.sub(r"(?<![\d.])(\d{1,3})((?: \d{3})+)(?![\d.])", lambda m: m.group(1) + m.group(2).replace(" ", ""), text)


def number_units(text: str) -> list[str]:
    """Canonical number + unit mentions in ``text``, in order: ``"500 mg"``, ``"3-5 day"``."""
    out = []
    for m in NUMBER_UNIT_RE.finditer(_prepare(text)):
        unit = _UNIT_CANON[m.group("unit")]
        out.append(f"{m.group('lo')}-{m.group('hi')} {unit}" if m.group("hi") else f"{m.group('lo')} {unit}")
    return out


def supported_set(text: str) -> set[str]:
    """Every canonical value in ``text``, plus the endpoints of each range."""
    values: set[str] = set()
    for v in number_units(text):
        values.add(v)
        num, unit = v.split(" ", 1)
        if "-" in num:
            lo, hi = num.split("-")
            values.update({f"{lo} {unit}", f"{hi} {unit}"})
    return values


def is_supported(value: str, supported: set[str]) -> bool:
    if value in supported:
        return True
    num, unit = value.split(" ", 1)
    if "-" in num:
        lo, hi = num.split("-")
        return f"{lo} {unit}" in supported and f"{hi} {unit}" in supported
    return False


def unsupported_numbers(answer: str, cited_records: list[dict[str, Any]], question: str = "") -> list[dict[str, Any]]:
    """Number + unit values in ``answer`` not found in any cited chunk's source text (deduplicated)."""
    supported: set[str] = set()
    for rec in cited_records:
        supported |= supported_set(verifiable_text(rec))
    in_question = supported_set(question)
    # bare numbers of the question count too: "systolic pressure ... about 134" -> "134 mmHg"
    bare_in_question = set(re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?!\.?\d)", _prepare(question)))
    flags, seen = [], set()
    for value in number_units(answer):
        if value in seen or is_supported(value, supported):
            continue
        seen.add(value)
        num = value.split(" ", 1)[0]
        flags.append({"value": value, "in_question": is_supported(value, in_question)
                      or all(p in bare_in_question for p in num.split("-"))})
    return flags


def clean_quote(quote: str) -> str:
    return quote.strip().strip(_QUOTE_MARKS).strip()


_TRAILING_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*$")


def quote_variants(quote: str) -> list[str]:
    """The cleaned quote first, then its tolerated forms (rules 2 and 3 in the module docstring)."""
    strict = clean_quote(quote)
    trimmed = _TRAILING_ELLIPSIS.sub("", strict)
    if trimmed.endswith("."):
        trimmed = trimmed[:-1].rstrip()
    out: list[str] = []
    for v in (strict, trimmed):
        for w in (v, v[:1].swapcase() + v[1:] if v[:1].isalpha() else v):
            if w and w not in out:
                out.append(w)
    return out


def quote_matches(quote: str, text: str) -> tuple[bool, bool]:
    """``(matches, tolerant)``: whether the quote occurs in ``text``, and whether that needed rule 2 or 3."""
    variants = quote_variants(quote)
    if not variants:
        return False, False
    if quote_in_text(variants[0], text):
        return True, False
    return (True, True) if any(quote_in_text(v, text) for v in variants[1:]) else (False, False)


def verify_citations(citations: list[dict[str, str]], sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per citation: ``{label, quote, chunk_id, status, verified, tolerant, correct_label, correct_chunk_id,
    reason}``.

    ``sources`` maps label -> chunk record, for every source shown to the model."""
    checks = []
    for c in citations:
        label, quote = c.get("label", ""), clean_quote(c.get("quote", ""))
        rec = sources.get(label)
        check: dict[str, Any] = {"label": label, "quote": c.get("quote", ""),
                                 "chunk_id": rec["chunk_id"] if rec else None, "status": "not_verified",
                                 "verified": False, "tolerant": False, "correct_label": None, "correct_chunk_id": None,
                                 "reason": None}
        checks.append(check)
        if not quote:
            check["reason"] = "empty quote"
            continue
        ok, tolerant = quote_matches(quote, verifiable_text(rec)) if rec else (False, False)
        if ok:
            check.update(status="verified", verified=True, tolerant=tolerant)
            continue
        for other_label, other in sources.items():
            ok, tolerant = quote_matches(quote, verifiable_text(other))
            if other_label != label and ok:
                check.update(status="verified_wrong_label", correct_label=other_label, correct_chunk_id=other["chunk_id"],
                             tolerant=tolerant,
                             reason=f"matches {other_label}, not {label}")
                break
        else:
            check["reason"] = "unknown label" if rec is None else "not in source text"
    return checks


LABEL_RE = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=\S)")
_LEADING_LABELS = re.compile(rf"^(?:\s*{LABEL_RE.pattern})+", re.IGNORECASE)


def answer_sentences(answer: str) -> list[str]:
    """Sentences of ``answer``; labels opening a sentence are given to the sentence before."""
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(answer.strip()):
        lead = _LEADING_LABELS.match(part)
        if lead and out:
            out[-1] += " " + lead.group(0).strip()
            part = part[lead.end():]
        if part.strip():
            out.append(part.strip())
    return out


def unlabelled_number_sentences(answer: str | None) -> list[str]:
    """Sentences with a digit (outside labels) and no ``[Sn]`` label."""
    return [s for s in answer_sentences(answer or "")
            if re.search(r"\d", LABEL_RE.sub("", s)) and not LABEL_RE.search(s)]


def check_result(result: dict[str, Any], chunks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Citation checks, number flags and unlabelled number sentences for an ``Answerer.answer``
    result (empty for refusals). Sources are what the model was shown (``result["sources"]``;
    ``result["retrieved"]`` for v1 results). For the number check a wrong-label citation counts
    for the source it actually matches."""
    shown = result.get("sources") or result["retrieved"]
    sources = {r["label"]: chunks[r["chunk_id"]] for r in shown}
    if not result.get("valid") or result.get("refused"):
        return {"citation_checks": [], "number_flags": [], "unlabelled_number_sentences": []}
    checks = verify_citations(result["citations"], sources)
    cited_labels = [c["correct_label"] or c["label"] for c in checks]
    cited = list({sources[lab]["chunk_id"]: sources[lab] for lab in cited_labels if lab in sources}.values())
    answer = result["answer"] or ""
    return {"citation_checks": checks,
            "number_flags": unsupported_numbers(answer, cited, result["question"]),
            "unlabelled_number_sentences": unlabelled_number_sentences(answer)}


# public names for the number parsing used by the metrics (src/evaluate/generation_metrics.py)
UNIT_CANON = _UNIT_CANON
prepare_numbers = _prepare
