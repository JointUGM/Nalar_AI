"""The parent summary, its guard and its template (design doc §7). Pure text, no model."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nalar_ai.shared.enums import ConceptOutcome
from nalar_ai.shared.text import contains_phrase, contains_stem, normalize_key
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import keys, without_names

_DIGIT = re.compile(r"\d")


@dataclass(frozen=True, slots=True)
class ParentConcept:
    name: str
    outcome: ConceptOutcome
    misconception: str | None
    resolved: bool


@dataclass(frozen=True, slots=True)
class ParentSummaryInput:
    mission_title: str
    concepts: tuple[ParentConcept, ...]
    evaluation_summary: str | None

    def names(self) -> tuple[str, ...]:
        """Teacher-written text the digit check exempts."""
        return (
            self.mission_title,
            *(c.name for c in self.concepts),
            *(c.misconception for c in self.concepts if c.misconception),
        )


@dataclass(frozen=True, slots=True)
class SummaryRules:
    """Terms as normalize_key() output. Build with `SummaryRules.build`."""

    min_words: int
    max_words: int
    banned_terms: tuple[str, ...]  # scores and integrity; affixed forms match too
    comparison_phrases: tuple[str, ...]  # whole-phrase match

    @classmethod
    def build(
        cls,
        *,
        min_words: int,
        max_words: int,
        banned_terms: Iterable[str],
        comparison_phrases: Iterable[str],
    ) -> "SummaryRules":
        return cls(min_words, max_words, keys(banned_terms), keys(comparison_phrases))


def content_problems(text: str, rules: SummaryRules, names: Iterable[str] = ()) -> list[str]:
    """Everything but length: the loader also runs this on the template."""
    text = without_names(text, names)
    problems: list[str] = []
    if _DIGIT.search(text):
        problems.append("write no digits or numbers")
    key = normalize_key(text)
    problems += [
        f"do not mention scores, cheating or copying (found '{term}')"
        for term in rules.banned_terms
        if contains_stem(key, term)
    ]
    problems += [
        f"do not compare Ananda with other students (found '{phrase}')"
        for phrase in rules.comparison_phrases
        if contains_phrase(key, phrase)
    ]
    return problems


def check_summary(text: str, rules: SummaryRules, names: Iterable[str] = ()) -> list[str]:
    """Problems phrased for the retry call. Empty means the summary may be stored."""
    problems: list[str] = []
    words = len(text.split())
    if words < rules.min_words:
        problems.append(f"write at least {rules.min_words} words")
    elif words > rules.max_words:
        problems.append(f"use at most {rules.max_words} words")
    return problems + content_problems(text, rules, names)


def join_names(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} dan {names[-1]}"


@dataclass(frozen=True, slots=True)
class SummaryTemplate:
    """Used when the model's summary fails the guard twice. Fields: {concepts}, {concept}."""

    opening: str
    understood: str
    developing: str
    home: str

    def render(self, concepts: Sequence[ParentConcept]) -> str:
        understood = [c.name for c in concepts if c.outcome is ConceptOutcome.MASTERED]
        developing = [c.name for c in concepts if c.outcome is not ConceptOutcome.MASTERED]
        parts = [self.opening]
        if understood:
            parts.append(self.understood.format(concepts=join_names(understood)))
        if developing:
            parts.append(self.developing.format(concepts=join_names(developing)))
        parts.append(self.home.format(concept=(developing or understood)[0]))
        return " ".join(parts)
