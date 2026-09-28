from dataclasses import replace
from uuid import UUID

import pytest

from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
    ModelLabels,
    normalize_labels,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType as T
from nalar_ai.subsystems.s3_socratic_prober.domain.prefilter import PrefilterHit
from tests.support.s3 import M_HABIS, M_KASAR

ANSWER = "Kelerengnya berhenti karena gayanya habis, capek aku"
LABELS = ModelLabels(
    answer_type=T.MISCONCEPTION,
    confident=True,
    misconception_ids=(M_HABIS,),
    frustration=True,
    safety_concern=False,
    key_phrase="gayanya habis",
)


def test_confident_labels_pass_through() -> None:
    result = normalize_labels(LABELS, ANSWER)
    assert result == Classification(
        T.MISCONCEPTION, ClassificationSource.MODEL, (M_HABIS,), True, "gayanya habis"
    )
    assert result.primary_misconception == M_HABIS and result.secondary_misconception is None


def test_low_confidence_becomes_unsure() -> None:
    result = normalize_labels(replace(LABELS, confident=False), ANSWER)
    assert (result.answer_type, result.misconception_ids) == (T.UNSURE, ())


def test_safety_concern_wins_over_everything() -> None:
    result = normalize_labels(replace(LABELS, safety_concern=True, confident=False), ANSWER)
    assert (result.answer_type, result.misconception_ids) == (T.SAFETY, ())


@pytest.mark.parametrize(
    ("answer_type", "ids", "expected_type", "expected_ids"),
    [
        (T.MISCONCEPTION, (), T.UNSURE, ()),
        (T.MIXED, (), T.UNSURE, ()),
        (T.CORRECT_REASONED, (M_HABIS,), T.UNSURE, ()),
        (T.EVASIVE, (M_HABIS,), T.EVASIVE, ()),
        (T.MANIPULATION, (M_HABIS,), T.MANIPULATION, ()),
        (T.MIXED, (M_HABIS, M_HABIS, M_KASAR, M_HABIS), T.MIXED, (M_HABIS, M_KASAR)),
    ],
)
def test_contradictions_are_resolved_safely(
    answer_type: T, ids: tuple[UUID, ...], expected_type: T, expected_ids: tuple[UUID, ...]
) -> None:
    result = normalize_labels(
        replace(LABELS, answer_type=answer_type, misconception_ids=ids), ANSWER
    )
    assert (result.answer_type, result.misconception_ids) == (expected_type, expected_ids)


@pytest.mark.parametrize(
    ("phrase", "kept"),
    [
        ("gayanya habis", "gayanya habis"),
        ("  Gayanya HABIS ", "Gayanya HABIS"),
        ("dorongannya hilang", None),  # not in the answer
        ("", None),
        ("gayanya habis " * 10, None),  # too long
    ],
)
def test_key_phrase_must_come_from_the_answer(phrase: str, kept: str | None) -> None:
    assert normalize_labels(replace(LABELS, key_phrase=phrase), ANSWER).key_phrase == kept


def test_fallback_and_prefilter_constructors() -> None:
    assert Classification.fallback() == Classification(T.UNSURE, ClassificationSource.FALLBACK)
    assert Classification.from_prefilter(PrefilterHit.MANIPULATION).answer_type is T.MANIPULATION
    assert Classification.from_prefilter(PrefilterHit.SAFETY).answer_type is T.SAFETY
