"""Text normalisation shared by all parsers and, later, by quote verification.

Two levels of normalisation are provided:

``normalise_text``
    Applied to every chunk before it is written. Unicode NFC, ligature
    expansion, invisible-character removal and whitespace tidying. Paragraph
    and list-item breaks (``\\n``) are preserved so chunks stay readable.

``normalise_for_match``
    Applied to *both* a quote and a chunk at verification time. On top of
    ``normalise_text`` it folds typographic variants (curly quotes, dashes)
    and collapses all whitespace, including newlines, to single spaces. An
    exact quote copied from a chunk therefore always matches, and a quote
    that differs only in quote style, dash style or line breaks still does.

Hyphenation at line ends is resolved while lines are joined
(:func:`join_lines`), because that decision needs the line structure.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

LIGATURES: dict[str, str] = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

# Characters that carry no visible content in running text.
_INVISIBLE_RE = re.compile("[­​‌‍⁠﻿]")
# Every horizontal whitespace variant, collapsed to a single ASCII space.
_HSPACE_RE = re.compile(r"[ \t  -   　]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

_MATCH_FOLDS: dict[str, str] = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    "…": "...",
}
_MATCH_TRANS = str.maketrans(_MATCH_FOLDS)

# A word fragment at the end of a line followed by a hyphen.
_TRAILING_HYPHEN_RE = re.compile(r"([A-Za-z]+)-$")
_LEADING_WORD_RE = re.compile(r"^([a-z]+)")
# An en dash or slash glued to the preceding character ("40–", "ACEis/"); a spaced
# parenthetical dash ("Hypertension –") is not a tight break.
_TIGHT_BREAK_RE = re.compile(r"\S[–/]$")
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def normalise_text(text: str) -> str:
    """Return ``text`` in the canonical form stored in chunk files.

    Steps: expand ligatures, Unicode NFC (so a base letter followed by a
    combining accent becomes one code point), remove soft hyphens and
    zero-width characters, collapse horizontal whitespace, trim spaces around
    line breaks and collapse blank lines. The function is idempotent.
    """
    for lig, repl in LIGATURES.items():
        text = text.replace(lig, repl)
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HSPACE_RE.sub(" ", text)
    text = "\n".join(part.strip() for part in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n", text)
    return text.strip()


def normalise_for_match(text: str) -> str:
    """Return the form used to compare quotes against chunks.

    Applies :func:`normalise_text`, folds curly quotes and dash variants to
    ASCII, and collapses all whitespace (including newlines) to single spaces.
    Case is preserved, because a change of case can change clinical meaning
    (e.g. drug names, units).
    """
    text = normalise_text(text).translate(_MATCH_TRANS)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str) -> bool:
    """True if ``quote`` occurs in ``text`` after both are match-normalised."""
    q = normalise_for_match(quote)
    return bool(q) and q in normalise_for_match(text)


def build_vocabulary(lines: Iterable[str]) -> set[str]:
    """Lower-cased words that appear *inside* lines (not split across a line end).

    Used by :func:`join_lines` to decide whether a line-end hyphen is a real
    compound hyphen ("cost-effective") or a typesetting break ("signi-ficant").
    Hyphenated compounds are included both whole and unhyphenated-whole so
    that either spelling can be detected.
    """
    vocab: set[str] = set()
    for line in lines:
        body = line.rstrip()
        if body.endswith("-"):
            body = body[: body.rfind(" ")] if " " in body else ""
        for word in WORD_RE.findall(body):
            vocab.add(word.lower())
    return vocab


def join_lines(lines: list[str], vocabulary: set[str] | None = None) -> str:
    """Join physical lines of one paragraph into a single string.

    Line-end handling:

    * ``word-`` + ``continuation`` keeps the hyphen ("cost-" + "effective" ->
      "cost-effective") **unless** the unhyphenated join is a word seen
      elsewhere in the document and the hyphenated form is not, in which case
      the hyphen is removed ("signi-" + "ficant" -> "significant"). Keeping the
      hyphen is the safe default: WHO typesetting rarely breaks words, but
      compounds are common.
    * A line ending in an en dash or slash glued to the previous character is
      joined without a space ("Telmisartan 40–" + "Amlodipine" ->
      "Telmisartan 40–Amlodipine"); a spaced dash ("Hypertension –") is not.
    * Otherwise lines are joined with a single space.
    """
    vocabulary = vocabulary or set()
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not out:
            out = line
            continue
        hyphen = _TRAILING_HYPHEN_RE.search(out)
        nxt = _LEADING_WORD_RE.match(line)
        if hyphen and nxt:
            head, tail = hyphen.group(1), nxt.group(1)
            joined = (head + tail).lower()
            hyphenated = f"{head}-{tail}".lower()
            if joined in vocabulary and hyphenated not in vocabulary:
                out = out[:-1] + line
            else:
                out = out + line
        elif _TIGHT_BREAK_RE.search(out):
            out = out + line
        else:
            out = out + " " + line
    return out
