"""Step 1: rules that decide simple answers without a model (design doc §6)."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from nalar_ai.shared.text import contains_phrase, normalize_key


class PrefilterHit(StrEnum):
    SAFETY = "safety"
    EVASIVE = "evasive"
    MANIPULATION = "manipulation"


@dataclass(frozen=True, slots=True)
class PrefilterLexicon:
    """Phrases as normalize_key() output. Build with `PrefilterLexicon.build`."""

    safety_phrases: tuple[str, ...]
    evasive_answers: frozenset[str]
    manipulation_phrases: tuple[str, ...]
    max_evasive_chars: int
    max_manipulation_words: int

    @classmethod
    def build(
        cls,
        *,
        safety_phrases: Iterable[str],
        evasive_answers: Iterable[str],
        manipulation_phrases: Iterable[str],
        max_evasive_chars: int,
        max_manipulation_words: int,
    ) -> "PrefilterLexicon":
        return cls(
            safety_phrases=_keys(safety_phrases),
            evasive_answers=frozenset(_keys(evasive_answers)),
            manipulation_phrases=_keys(manipulation_phrases),
            max_evasive_chars=max_evasive_chars,
            max_manipulation_words=max_manipulation_words,
        )


def prefilter(answer: str, lexicon: PrefilterLexicon) -> PrefilterHit | None:
    """Safety always wins. Manipulation only short-circuits short answers: a long answer
    that also asks for the answer still carries reasoning, so the classifier reads it."""
    key = normalize_key(answer)
    if any(contains_phrase(key, phrase) for phrase in lexicon.safety_phrases):
        return PrefilterHit.SAFETY
    if len(key) <= lexicon.max_evasive_chars or key in lexicon.evasive_answers:
        return PrefilterHit.EVASIVE
    if len(key.split()) <= lexicon.max_manipulation_words and any(
        contains_phrase(key, phrase) for phrase in lexicon.manipulation_phrases
    ):
        return PrefilterHit.MANIPULATION
    return None


def _keys(phrases: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for key in (normalize_key(p) for p in phrases) if key))
