"""Schema for evaluation items (``eval/questions.jsonl``, one JSON object per line).

An item is a question with its expected behaviour and, when it is answerable, the gold
answer and the verbatim source spans that support it::

    {
      "qid": "...", "question": "...", "category": "dose_lookup", "collection": "aware",
      "answerable": true, "expected_behaviour": "answer", "gold_answer": "...",
      "gold_sources": [{"doc_id": "who-aware-2022", "printed_page": 290, "span": "..."}],
      "gold_structured": {"any_of": [{"drug": "...", "dose": "...", "frequency": "...", "duration": "..."}]},
      "notes": "..."
    }

Rules checked by :func:`validate_item` (structure only; whether the spans really occur in
the chunk files is checked by ``scripts/validate_eval.py``):

* every key above is present (``gold_answer``, ``gold_structured``, ``collection`` and
  ``notes`` may be null) and no other key is;
* ``category`` and ``expected_behaviour`` take one of the listed values;
* ``answerable`` items expect ``answer`` and have a ``collection``, a ``gold_answer`` and at
  least one gold source; unanswerable items expect ``decline`` or ``redirect`` and have **empty**
  ``gold_sources`` and no ``gold_answer`` or ``gold_structured``;
* a gold source is ``{doc_id, printed_page, span}`` with a non-empty span and an integer
  printed page; ``gold_structured`` is ``{"any_of": [...]}`` with at least one option, each
  option holding ``drug``, ``dose``, ``frequency`` and ``duration`` (strings or null; ``drug``
  required). ``any_of`` lists the acceptable alternatives (e.g. equal first-line options).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CATEGORIES: tuple[str, ...] = (
    "single_rec",
    "combined",
    "dose_lookup",
    "population_variant",
    "no_antibiotic",
    "conflicting_source",
    "near_miss",
    "out_of_domain",
    "personal_advice",
)
BEHAVIOURS: tuple[str, ...] = ("answer", "decline", "redirect")

ITEM_KEYS: tuple[str, ...] = (
    "qid", "question", "category", "collection", "answerable", "expected_behaviour", "gold_answer",
    "gold_sources", "gold_structured", "notes",
)
SOURCE_KEYS: tuple[str, ...] = ("doc_id", "printed_page", "span")
STRUCTURED_KEYS: tuple[str, ...] = ("drug", "dose", "frequency", "duration")


class EvalItemError(ValueError):
    """Raised by :func:`parse_item` when an item breaks the schema; lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class GoldSource:
    doc_id: str
    printed_page: int
    span: str  # verbatim source text (matched after text_utils.normalise_for_match)


@dataclass(frozen=True)
class StructuredOption:
    drug: str
    dose: str | None = None
    frequency: str | None = None
    duration: str | None = None


@dataclass(frozen=True)
class GoldStructured:
    any_of: tuple[StructuredOption, ...]


@dataclass(frozen=True)
class EvalItem:
    qid: str
    question: str
    category: str
    collection: str | None
    answerable: bool
    expected_behaviour: str
    gold_answer: str | None
    gold_sources: tuple[GoldSource, ...] = field(default_factory=tuple)
    gold_structured: GoldStructured | None = None
    notes: str | None = None


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_text(value: Any) -> bool:
    return value is None or _is_text(value)


def _keys_errors(obj: dict[str, Any], expected: tuple[str, ...], where: str) -> list[str]:
    errors = []
    missing = [k for k in expected if k not in obj]
    extra = [k for k in obj if k not in expected]
    if missing:
        errors.append(f"{where}: missing keys {missing}")
    if extra:
        errors.append(f"{where}: unknown keys {extra}")
    return errors


def validate_item(obj: Any) -> list[str]:
    """Return every schema violation of one raw item (empty if it is valid)."""
    if not isinstance(obj, dict):
        return ["item is not a JSON object"]
    errors = _keys_errors(obj, ITEM_KEYS, "item")
    if errors:
        return errors
    for key in ("qid", "question"):
        if not _is_text(obj[key]):
            errors.append(f"{key} must be a non-empty string")
    for key in ("collection", "gold_answer", "notes"):
        if not _optional_text(obj[key]):
            errors.append(f"{key} must be a non-empty string or null")
    if obj["category"] not in CATEGORIES:
        errors.append(f"category {obj['category']!r} not in {list(CATEGORIES)}")
    if obj["expected_behaviour"] not in BEHAVIOURS:
        errors.append(f"expected_behaviour {obj['expected_behaviour']!r} not in {list(BEHAVIOURS)}")
    if not isinstance(obj["answerable"], bool):
        errors.append("answerable must be true or false")

    sources = obj["gold_sources"]
    if not isinstance(sources, list):
        errors.append("gold_sources must be a list")
        sources = []
    for i, src in enumerate(sources):
        where = f"gold_sources[{i}]"
        if not isinstance(src, dict):
            errors.append(f"{where}: not an object")
            continue
        key_errors = _keys_errors(src, SOURCE_KEYS, where)
        errors.extend(key_errors)
        if key_errors:
            continue
        if not _is_text(src["doc_id"]):
            errors.append(f"{where}: doc_id must be a non-empty string")
        if not isinstance(src["printed_page"], int) or isinstance(src["printed_page"], bool):
            errors.append(f"{where}: printed_page must be an integer")
        if not _is_text(src["span"]):
            errors.append(f"{where}: span must be a non-empty string")

    structured = obj["gold_structured"]
    if structured is not None:
        if not isinstance(structured, dict) or list(structured) != ["any_of"]:
            errors.append('gold_structured must be null or {"any_of": [...]}')
        elif not isinstance(structured["any_of"], list) or not structured["any_of"]:
            errors.append("gold_structured.any_of must be a non-empty list")
        else:
            for i, opt in enumerate(structured["any_of"]):
                where = f"gold_structured.any_of[{i}]"
                if not isinstance(opt, dict):
                    errors.append(f"{where}: not an object")
                    continue
                key_errors = _keys_errors(opt, STRUCTURED_KEYS, where)
                errors.extend(key_errors)
                if key_errors:
                    continue
                if not _is_text(opt["drug"]):
                    errors.append(f"{where}: drug must be a non-empty string")
                for key in ("dose", "frequency", "duration"):
                    if not _optional_text(opt[key]):
                        errors.append(f"{where}: {key} must be a non-empty string or null")

    if obj["answerable"] is True:
        if obj["expected_behaviour"] != "answer":
            errors.append("answerable items must have expected_behaviour 'answer'")
        if not sources:
            errors.append("answerable items need at least one gold source")
        if obj["gold_answer"] is None:
            errors.append("answerable items need a gold_answer")
        if obj["collection"] is None:
            errors.append("answerable items need a collection")
    elif obj["answerable"] is False:
        if obj["expected_behaviour"] == "answer":
            errors.append("unanswerable items must have expected_behaviour 'decline' or 'redirect'")
        if sources:
            errors.append("unanswerable items must have empty gold_sources")
        if obj["gold_answer"] is not None or structured is not None:
            errors.append("unanswerable items must have null gold_answer and gold_structured")
    return errors


def parse_item(obj: Any) -> EvalItem:
    """Build an :class:`EvalItem` from a raw item, raising :class:`EvalItemError` if invalid."""
    errors = validate_item(obj)
    if errors:
        raise EvalItemError(errors)
    structured = obj["gold_structured"]
    return EvalItem(
        qid=obj["qid"],
        question=obj["question"],
        category=obj["category"],
        collection=obj["collection"],
        answerable=obj["answerable"],
        expected_behaviour=obj["expected_behaviour"],
        gold_answer=obj["gold_answer"],
        gold_sources=tuple(GoldSource(**src) for src in obj["gold_sources"]),
        gold_structured=GoldStructured(tuple(StructuredOption(**opt) for opt in structured["any_of"]))
        if structured is not None else None,
        notes=obj["notes"],
    )
