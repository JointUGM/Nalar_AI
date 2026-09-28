"""Step 2 in code: turn the classifier's labels into a safe classification (design doc §7)."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from nalar_ai.shared.text import contains_phrase, normalize_key
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import MISCONCEPTION_TYPES, AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.prefilter import PrefilterHit

MAX_MISCONCEPTIONS = 2
MAX_KEY_PHRASE_CHARS = 80

_CORRECT_TYPES = frozenset({AnswerType.CORRECT_REASONED, AnswerType.CORRECT_UNREASONED})
_NO_MISCONCEPTION_TYPES = frozenset({AnswerType.EVASIVE, AnswerType.MANIPULATION})


class ClassificationSource(StrEnum):
    PREFILTER = "prefilter"
    MODEL = "model"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class Classification:
    answer_type: AnswerType
    source: ClassificationSource
    misconception_ids: tuple[UUID, ...] = ()
    frustration: bool = False
    key_phrase: str | None = None

    @property
    def primary_misconception(self) -> UUID | None:
        return self.misconception_ids[0] if self.misconception_ids else None

    @property
    def secondary_misconception(self) -> UUID | None:
        return self.misconception_ids[1] if len(self.misconception_ids) > 1 else None

    @classmethod
    def fallback(cls) -> "Classification":
        """Asking "why?" is never wrong: used whenever the classifier fails or is slow."""
        return cls(AnswerType.UNSURE, ClassificationSource.FALLBACK)

    @classmethod
    def from_prefilter(cls, hit: PrefilterHit) -> "Classification":
        return cls(AnswerType(hit.value), ClassificationSource.PREFILTER)


@dataclass(frozen=True, slots=True)
class ModelLabels:
    """The classifier's output with aliases already resolved (unknown aliases dropped)."""

    answer_type: AnswerType
    confident: bool
    misconception_ids: tuple[UUID, ...]
    frustration: bool
    safety_concern: bool
    key_phrase: str


def normalize_labels(labels: ModelLabels, answer: str) -> Classification:
    ids = tuple(dict.fromkeys(labels.misconception_ids))[:MAX_MISCONCEPTIONS]
    answer_type = labels.answer_type
    if labels.safety_concern:
        answer_type, ids = AnswerType.SAFETY, ()
    elif not labels.confident:
        answer_type, ids = AnswerType.UNSURE, ()
    elif answer_type in MISCONCEPTION_TYPES and not ids:
        answer_type = AnswerType.UNSURE
    elif answer_type in _CORRECT_TYPES and ids:
        answer_type, ids = AnswerType.UNSURE, ()  # contradictory labels
    elif answer_type in _NO_MISCONCEPTION_TYPES:
        ids = ()
    return Classification(
        answer_type=answer_type,
        source=ClassificationSource.MODEL,
        misconception_ids=ids,
        frustration=labels.frustration,
        key_phrase=_key_phrase(labels.key_phrase, answer),
    )


def _key_phrase(phrase: str, answer: str) -> str | None:
    phrase = phrase.strip()
    if not phrase or len(phrase) > MAX_KEY_PHRASE_CHARS:
        return None
    return phrase if contains_phrase(normalize_key(answer), normalize_key(phrase)) else None
