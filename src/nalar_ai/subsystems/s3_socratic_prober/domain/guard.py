"""Step 7: the leak guard (AI-9, design doc §11). Cheap text checks first, then similarity."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from nalar_ai.shared.enums import GuardResult
from nalar_ai.shared.text import contains_phrase, normalize_key, stem_spans
from nalar_ai.shared.vectors import cosine

_SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)")


@dataclass(frozen=True, slots=True)
class GuardLexicon:
    """Verdict terms as normalize_key() output. Build with `GuardLexicon.build`."""

    verdict_terms: tuple[str, ...]
    max_chars: int
    max_sentences: int

    @classmethod
    def build(
        cls, *, verdict_terms: Iterable[str], max_chars: int, max_sentences: int
    ) -> "GuardLexicon":
        keys = (normalize_key(term) for term in verdict_terms)
        return cls(tuple(dict.fromkeys(k for k in keys if k)), max_chars, max_sentences)


@dataclass(frozen=True, slots=True)
class GuardThresholds:
    max_reference_similarity: float
    min_approved_similarity: float


def count_sentences(text: str) -> int:
    return len([part for part in _SENTENCE_END.split(text.strip()) if part.strip()])


def check_text(
    adapted: str,
    approved: str,
    student_answers: Sequence[str],
    answer_terms: Sequence[str],
    lexicon: GuardLexicon,
) -> GuardResult | None:
    """None means the text checks passed. Terms the approved question or the student already
    used are allowed: repeating the student's own words is the point of adapting."""
    text = adapted.strip()
    if (
        not text
        or len(text) > lexicon.max_chars
        or not text.endswith("?")
        or count_sentences(text) > lexicon.max_sentences
    ):
        return GuardResult.BLOCKED_SHAPE
    adapted_key = normalize_key(text)
    approved_key = normalize_key(approved)
    for term in lexicon.verdict_terms:
        if contains_phrase(adapted_key, term) and not contains_phrase(approved_key, term):
            return GuardResult.BLOCKED_VERDICT
    known = [approved_key, *(normalize_key(answer) for answer in student_answers)]

    def already_used(form: str) -> bool:
        return any(contains_phrase(source, form) for source in known)

    for term in answer_terms:
        term_key = normalize_key(term)
        # Stems, not whole words: "gesekannya" and "bergesekan" still give "gesekan" away.
        spans = stem_spans(adapted_key, term_key)
        # Known only as the term itself or the very word form used here: a stem inside another
        # word of a source ("harus" for "arus") is not proof the idea was already named.
        if spans and not (already_used(term_key) or all(already_used(s) for s in spans)):
            return GuardResult.BLOCKED_NEW_TERMS
    return None


def check_similarity(
    adapted: Sequence[float],
    approved: Sequence[float],
    reference: Sequence[float],
    thresholds: GuardThresholds,
) -> GuardResult:
    to_reference = cosine(adapted, reference)
    if to_reference >= thresholds.max_reference_similarity and to_reference > cosine(
        approved, reference
    ):
        return GuardResult.BLOCKED_SIMILARITY
    if cosine(adapted, approved) < thresholds.min_approved_similarity:
        return GuardResult.BLOCKED_DRIFT
    return GuardResult.PASSED
