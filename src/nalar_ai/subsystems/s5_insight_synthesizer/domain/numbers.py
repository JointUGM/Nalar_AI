"""The AI-4 number check (design doc §6): model text never states a count."""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from nalar_ai.shared.text import contains_phrase, normalize_key

_DIGIT = re.compile(r"\d")


def keys(terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for key in map(normalize_key, terms) if key))


@dataclass(frozen=True, slots=True)
class NumberLexicon:
    """Terms as normalize_key() output, matched as whole words. Build with `build`."""

    number_words: tuple[str, ...]
    count_claims: tuple[str, ...]

    @classmethod
    def build(cls, *, number_words: Iterable[str], count_claims: Iterable[str]) -> "NumberLexicon":
        return cls(keys(number_words), keys(count_claims))


def without_names(text: str, names: Iterable[str]) -> str:
    """Remove provided names (title, concepts, statements): "Hukum Newton 1" is not a count."""
    for name in sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True):
        text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    return text


def number_problems(text: str, lexicon: NumberLexicon, names: Iterable[str] = ()) -> list[str]:
    """Problems phrased for the retry call. Empty means no number is written."""
    text = without_names(text, names)
    problems: list[str] = []
    if _DIGIT.search(text):
        problems.append("write no digits")
    key = normalize_key(text)
    problems += [
        f"write no number words (found '{word}')"
        for word in lexicon.number_words
        if contains_phrase(key, word)
    ]
    problems += [
        f"do not say how many students with words (found '{claim}'); use a placeholder"
        for claim in lexicon.count_claims
        if contains_phrase(key, claim)
    ]
    return problems
