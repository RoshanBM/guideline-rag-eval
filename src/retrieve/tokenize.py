"""Tokenisation for lexical retrieval (BM25).

Rules, applied in order:

1. Lower-case; ``>=``/``<=`` become ``≥``/``≤``; negative contractions are expanded
   (``don't`` -> ``do not``, ``can't`` -> ``can not``, ``won't`` -> ``will not``) so the
   negation survives.
2. Split on non-alphanumerics (Unicode-aware; underscores split too). Decimal numbers stay
   whole (``0.9`` and ``6.5`` are one token each). Ranges split: ``3–5`` -> ``3``, ``5``.
3. A number glued to a unit is split: ``500mg`` -> ``500``, ``mg`` (same tokens as ``500 mg``).
   Other letter-digit tokens are kept whole (``hba1c``, ``covid19``).
4. A comparison sign before a number gives the signed token **and** the bare number:
   ``≥130`` (or ``≥ 130``) -> ``≥130``, ``130``. A trailing plus does the same:
   ``2+`` -> ``2+``, ``2``.
5. English stopwords (NLTK's list) are removed, **except** the negations ``no``, ``nor`` and
   ``not``. Numbers are never removed.
6. With ``stem=True`` (the default, used by the ``bm25`` baseline) the remaining word tokens
   are reduced with the Snowball English stemmer (PyStemmer, the stemmer ``bm25s`` uses):
   ``antibiotics`` -> ``antibiot``, ``days`` -> ``day``. Numbers, signed numbers (``≥130``,
   ``2+``) and the negations are never stemmed. ``stem=False`` is the unstemmed ablation
   (``bm25-nostem``).
"""

from __future__ import annotations

import re

import Stemmer

# Bump when the rules change: a saved index records the version it was built with.
TOKENIZER_VERSION = "2"
_STEMMER = Stemmer.Stemmer("english")

NEGATIONS: frozenset[str] = frozenset({"no", "nor", "not"})

# NLTK English stopword list (nltk.corpus.stopwords.words("english")), copied to avoid the
# dependency; the negations are removed from it below.
_NLTK_ENGLISH = """
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself
yourselves he him his himself she she's her hers herself it it's its itself they them their
theirs themselves what which who whom this that that'll these those am is are was were be been
being have has had having do does did doing a an the and but if or because as until while of at
by for with about against between into through during before after above below to from up down
in out on off over under again further then once here there when where why how all any both
each few more most other some such no nor not only own same so than too very s t can will just
don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't
needn needn't shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split()
STOPWORDS: frozenset[str] = frozenset(
    w for w in _NLTK_ENGLISH if "'" not in w and w not in NEGATIONS
)

# Units that are split off a number they are glued to ("500mg" -> "500", "mg").
UNITS: frozenset[str] = frozenset({
    "mg", "g", "kg", "mcg", "µg", "μg", "ng", "ml", "l", "dl", "mmhg", "mmol", "iu", "units",
    "h", "hr", "hrs", "hours", "min", "mins", "d", "days", "wk", "wks", "weeks", "y", "yr", "yrs",
    "years", "x",
})

_IRREGULAR_NEG = (
    (re.compile(r"\bcan['’]t\b"), "can not"),
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"\bshan['’]t\b"), "shall not"),
)
_NEG_CONTRACTION = re.compile(r"n['’]t\b")
_TOKEN = re.compile(
    r"(?P<cmp>[≥≤<>])?\s?"
    r"(?P<word>\d+(?:\.\d+)+[^\W_]*|[^\W_]+)"
    r"(?P<plus>\+)?"
)
_NUMBER = re.compile(r"\d+(?:\.\d+)*")
_NUMBER_UNIT = re.compile(r"(\d+(?:\.\d+)*)([^\W\d_]+)")


def tokenize(text: str, stem: bool = True) -> list[str]:
    """Tokens of ``text`` for BM25 indexing and querying (see the module docstring)."""
    tokens = _tokenize_unstemmed(text)
    if not stem:
        return tokens
    return [
        tok if (tok[0].isdigit() or tok[0] in "≥≤<>" or tok in NEGATIONS) else _STEMMER.stemWord(tok)
        for tok in tokens
    ]


def _tokenize_unstemmed(text: str) -> list[str]:
    text = text.lower().replace(">=", "≥").replace("<=", "≤")
    for pattern, repl in _IRREGULAR_NEG:
        text = pattern.sub(repl, text)
    text = _NEG_CONTRACTION.sub(" not", text)

    tokens: list[str] = []
    for m in _TOKEN.finditer(text):
        word, cmp, plus = m.group("word"), m.group("cmp"), m.group("plus")
        unit_split = _NUMBER_UNIT.fullmatch(word)
        if unit_split and unit_split.group(2) in UNITS:
            number, unit = unit_split.groups()
            if cmp:
                tokens.append(cmp + number)
            tokens.append(number)
            if unit not in STOPWORDS:  # "5d" -> "5", like "5 d" ("d" is a stopword)
                tokens.append(unit)
            continue
        if _NUMBER.fullmatch(word):
            if cmp:
                tokens.append(cmp + word)
            if plus:
                tokens.append(word + "+")
            tokens.append(word)
            continue
        if word not in STOPWORDS:
            tokens.append(word)
    return tokens
