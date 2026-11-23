from __future__ import annotations

import re
import unicodedata

STOPWORDS = frozenset(
    """a about after again all also am an and any are as at be been before being but by can could did do
    does doing for from get got had has have how i if in into is it its just me more most my no not of
    on once only or other our out over own please same should so some such than that the their them then
    there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours""".split()
)

_SUFFIXES = ("ations", "ation", "ables", "able", "ings", "ing", "als", "al", "edly", "ed", "ies", "s")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_COMPOUND_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


def fold(text: str) -> str:
    """Lower-case and strip accents: 'Réservé' -> 'reserve'."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def stem(word: str) -> str:
    if word.isdigit() or len(word) <= 3:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    if word.endswith("e") and len(word) > 4:  # "reserve" and "reserved" both become "reserv"
        word = word[:-1]
    return word


def words(text: str) -> list[str]:
    folded = fold(text)
    tokens = [w for w in _TOKEN_RE.findall(folded) if w not in STOPWORDS]
    tokens.extend(m.replace("-", "") for m in _COMPOUND_RE.findall(folded))
    return tokens


def tokenize(text: str) -> list[str]:
    return [stem(w) for w in words(text)]
