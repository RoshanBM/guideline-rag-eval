"""The single chunk schema shared by every collection (hypertension, AWaRe, ...).

Every line of every ``data/chunks/*.jsonl`` file is a record with exactly the
fields in ``FIELDS``, in that order. :func:`build_record` creates records in
that order, :func:`validate_record` checks them, and :func:`verifiable_text`
returns the part of a chunk that quotes may be checked against.

Text fields
-----------
``source_text``
    Verbatim source content only (normalised with ``text_utils.normalise_text``;
    formatting added by the parsers is limited to line breaks, ``" | "`` between
    table cells and ``"• "`` for list items).
``text``
    The retrieval text, assembled as::

        [context_header "\\n"] source_text ["\\nNotes:\\n" footnotes] ["\\n" aware_category_note]

    The footnotes are verbatim source text (table footnotes and table notes that
    the chunk uses). ``context_header`` and ``aware_category_note`` are written by
    the parser and are *not* source text.
``aware_category_note``
    AWaRe only. The Access/Watch/Reserve category of each antibiotic in the chunk.
    The source shows the category only by colour; this note is derived from that
    colour coding (``back_verde`` = ACCESS, ``back_amarelo`` = WATCH,
    ``back_vermelho`` = RESERVE) and is never used to verify quotes.

Quote verification must use :func:`verifiable_text` (source text plus footnotes),
never ``text``, so that a quote of a parser-written header or category note is
not accepted as a quote of the guideline.
"""

from __future__ import annotations

from typing import Any

from src.parse.text_utils import normalise_text, quote_in_text

NOTES_LABEL = "Notes:"

# (field, allowed JSON types). "str?" means string or null, etc.
FIELDS: list[tuple[str, str]] = [
    ("chunk_id", "str"),
    ("collection", "str"),
    ("doc", "str"),
    ("doc_id", "str"),
    ("section_path", "list[str]"),
    ("infection", "str?"),
    ("population", "str?"),
    ("population_raw", "str?"),
    ("context_header", "str?"),
    ("rec_id", "str?"),
    ("rec_part", "int?"),
    ("parent_rec_id", "str?"),
    ("chunk_type", "str"),
    ("index_default", "bool"),
    ("strength", "str?"),
    ("strength_as_printed", "str?"),
    ("certainty", "str?"),
    ("source_discrepancy", "str?"),
    ("source_text", "str"),
    ("aware_category_note", "str?"),
    ("text", "str"),
    ("pdf_page_start", "int"),
    ("pdf_page_end", "int"),
    ("printed_page_start", "int?"),
    ("printed_page_end", "int?"),
    ("source_url", "str"),
    ("licence", "str"),
]
FIELD_NAMES: list[str] = [name for name, _ in FIELDS]

CHUNK_TYPES = {"recommendation", "prose", "table", "pico", "no_antibiotic"}
POPULATIONS = {"adults", "adults_adolescents", "children", "neonates", "pregnant", "all"}
STRENGTHS = {"strong", "conditional"}


def build_record(**values: Any) -> dict[str, Any]:
    """Return a record with every schema field, in schema order (missing fields are null).

    Raises ``KeyError`` for an unknown field so that typos cannot slip through.
    """
    unknown = set(values) - set(FIELD_NAMES)
    if unknown:
        raise KeyError(f"Unknown chunk fields: {sorted(unknown)}")
    return {name: values.get(name) for name in FIELD_NAMES}


def compose_text(source_text: str, context_header: str | None = None, notes: list[str] | None = None,
                 aware_category_note: str | None = None) -> str:
    """Assemble the retrieval ``text`` from its parts (see module docstring)."""
    parts = []
    if context_header:
        parts.append(context_header)
    parts.append(source_text)
    if notes:
        parts.append(NOTES_LABEL + "\n" + "\n".join(notes))
    if aware_category_note:
        parts.append(aware_category_note)
    return normalise_text("\n".join(parts))


def split_text(record: dict[str, Any]) -> tuple[str | None, str, list[str], str | None]:
    """Split ``text`` back into (context_header, source_text, footnotes, aware_category_note).

    Raises ``ValueError`` if ``text`` is not assembled as documented.
    """
    text = record["text"]
    header, note, source = record["context_header"], record["aware_category_note"], record["source_text"]
    if header:
        if not text.startswith(header + "\n"):
            raise ValueError("text does not start with context_header")
        text = text[len(header) + 1 :]
    if note:
        if not text.endswith("\n" + note):
            raise ValueError("text does not end with aware_category_note")
        text = text[: -len(note) - 1]
    if not text.startswith(source):
        raise ValueError("source_text is not the body of text")
    rest = text[len(source) :]
    notes: list[str] = []
    if rest:
        prefix = "\n" + NOTES_LABEL + "\n"
        if not rest.startswith(prefix):
            raise ValueError(f"unexpected text after source_text: {rest[:60]!r}")
        notes = rest[len(prefix) :].split("\n")
    return header, source, notes, note


def verifiable_text(record: dict[str, Any]) -> str:
    """Source text plus footnotes: the only text a quote may be verified against."""
    _, source, notes, _ = split_text(record)
    return "\n".join([source, *notes])


def quote_verifies(quote: str, record: dict[str, Any]) -> bool:
    """True if ``quote`` occurs in the chunk's verifiable source text (see :func:`verifiable_text`)."""
    return quote_in_text(quote, verifiable_text(record))


def _type_ok(value: Any, spec: str) -> bool:
    optional = spec.endswith("?")
    base = spec.rstrip("?")
    if value is None:
        return optional
    if base == "str":
        return isinstance(value, str)
    if base == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if base == "bool":
        return isinstance(value, bool)
    if base == "list[str]":
        return isinstance(value, list) and all(isinstance(v, str) and v for v in value) and bool(value)
    raise ValueError(f"unknown type spec {spec}")


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return a list of schema violations for one record (empty if valid)."""
    errors: list[str] = []
    if list(record) != FIELD_NAMES:
        missing = [f for f in FIELD_NAMES if f not in record]
        extra = [f for f in record if f not in FIELD_NAMES]
        errors.append(f"fields differ from schema (missing {missing}, extra {extra}, or wrong order)")
        return errors
    for name, spec in FIELDS:
        if not _type_ok(record[name], spec):
            errors.append(f"{name}: {record[name]!r} is not {spec}")
    if errors:
        return errors
    if record["chunk_type"] not in CHUNK_TYPES:
        errors.append(f"chunk_type {record['chunk_type']!r} not in {sorted(CHUNK_TYPES)}")
    if record["population"] is not None and record["population"] not in POPULATIONS:
        errors.append(f"population {record['population']!r} not in {sorted(POPULATIONS)}")
    if (record["population"] is None) != (record["population_raw"] is None):
        errors.append("population and population_raw must be both null or both set")
    for name in ("strength", "strength_as_printed"):
        if record[name] is not None and record[name] not in STRENGTHS:
            errors.append(f"{name} {record[name]!r} not in {sorted(STRENGTHS)}")
    if record["pdf_page_start"] > record["pdf_page_end"]:
        errors.append("pdf_page_start > pdf_page_end")
    if not record["source_text"].strip():
        errors.append("source_text is empty")
    if normalise_text(record["text"]) != record["text"] or normalise_text(record["source_text"]) != record["source_text"]:
        errors.append("text/source_text not normalised")
    try:
        split_text(record)
    except ValueError as exc:
        errors.append(f"text structure: {exc}")
    return errors
