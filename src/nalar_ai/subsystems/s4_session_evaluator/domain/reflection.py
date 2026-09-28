"""The student's reflection and its guard (design doc §8, §9). Pure text checks, no model."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nalar_ai.shared.text import contains_phrase, contains_stem, normalize_key, stem_spans

_DIGIT = re.compile(r"\d")


@dataclass(frozen=True, slots=True)
class ReflectionParts:
    strengths: str
    changed_mind: str
    question: str

    @property
    def content(self) -> str:
        """session_reflections.content: three short paragraphs."""
        return "\n\n".join(
            part.strip() for part in (self.strengths, self.changed_mind, self.question)
        )


@dataclass(frozen=True, slots=True)
class ReflectionLexicon:
    """Terms as normalize_key() output. Build with `ReflectionLexicon.build`."""

    verdict_terms: tuple[str, ...]  # whole-word match
    banned_terms: tuple[str, ...]  # scores and integrity; affixed forms match too
    max_words: int

    @classmethod
    def build(
        cls, *, verdict_terms: Iterable[str], banned_terms: Iterable[str], max_words: int
    ) -> "ReflectionLexicon":
        return cls(_keys(verdict_terms), _keys(banned_terms), max_words)


def _keys(terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for key in map(normalize_key, terms) if key))


def check_reflection(
    parts: ReflectionParts,
    student_answers: Sequence[str],
    answer_terms: Sequence[str],
    lexicon: ReflectionLexicon,
) -> list[str]:
    """Problems with a reflection, phrased for the retry call. Empty means it may be shown."""
    problems: list[str] = []
    if not all(p.strip() for p in (parts.strengths, parts.changed_mind, parts.question)):
        problems.append("every part must be written")
    if not parts.question.strip().endswith("?"):
        problems.append("the question must end with '?'")
    content = parts.content
    if len(content.split()) > lexicon.max_words:
        problems.append(f"use at most {lexicon.max_words} words in total")
    if _DIGIT.search(content):
        problems.append("write no digits or numbers")
    key = normalize_key(content)
    for term in lexicon.verdict_terms:
        if contains_phrase(key, term):
            problems.append(f"do not judge right or wrong (found '{term}')")
    for term in lexicon.banned_terms:
        if contains_stem(key, term):
            problems.append(f"do not mention scores, cheating or copying (found '{term}')")
    own_words = [normalize_key(answer) for answer in student_answers]

    def student_wrote(form: str) -> bool:
        return any(contains_phrase(answer, form) for answer in own_words)

    for term in answer_terms:
        term_key = normalize_key(term)
        spans = stem_spans(key, term_key)
        # Exempt only when the student wrote the term itself or the very word form used here:
        # a stem match in their answer ("harus" for "arus") is not proof they named the idea.
        if spans and not (student_wrote(term_key) or all(student_wrote(s) for s in spans)):
            # Never echo the term: this text goes back to the writer, who must not learn it.
            problems.append("do not name a science idea or term the student did not write")
            break
    return problems
