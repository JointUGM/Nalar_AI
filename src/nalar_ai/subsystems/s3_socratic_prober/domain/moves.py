"""Answer types and the move table (PRD §7.2, design doc §8)."""

from collections.abc import Mapping
from enum import StrEnum

from nalar_ai.shared.enums import ProbeStrategy


class AnswerType(StrEnum):
    """The prober's label for one answer. The backend maps it onto answer_state (Q-S3-1)."""

    CORRECT_REASONED = "correct_reasoned"
    CORRECT_UNREASONED = "correct_unreasoned"
    MISCONCEPTION = "misconception"
    MIXED = "mixed"
    EVASIVE = "evasive"
    MANIPULATION = "manipulation"
    UNSURE = "unsure"  # set by code: low confidence, contradictions, failed classification
    SAFETY = "safety"  # set by code: the session pauses; no move


# The labels the classifier may return. "unsure" and "safety" are decided by code.
MODEL_ANSWER_TYPES: tuple[AnswerType, ...] = (
    AnswerType.CORRECT_REASONED,
    AnswerType.CORRECT_UNREASONED,
    AnswerType.MISCONCEPTION,
    AnswerType.MIXED,
    AnswerType.EVASIVE,
    AnswerType.MANIPULATION,
)

# Answer types that carry a known wrong idea.
MISCONCEPTION_TYPES = frozenset({AnswerType.MISCONCEPTION, AnswerType.MIXED})

# A challenge probe is a counter-example or a transfer (PRD glossary).
CHALLENGE_MOVES = frozenset({ProbeStrategy.COUNTER_EXAMPLE, ProbeStrategy.TRANSFER})

# Allowed moves per answer type, default first (PRD §7.2 step 2).
MOVE_TABLE: Mapping[AnswerType, tuple[ProbeStrategy, ...]] = {
    AnswerType.CORRECT_REASONED: (ProbeStrategy.TRANSFER, ProbeStrategy.DEEPER_REASON),
    AnswerType.CORRECT_UNREASONED: (
        ProbeStrategy.REQUEST_JUSTIFICATION,
        ProbeStrategy.DECOMPOSE,
    ),
    AnswerType.MISCONCEPTION: (
        ProbeStrategy.COUNTER_EXAMPLE,
        ProbeStrategy.EXPLAIN_MECHANISM,
        ProbeStrategy.DECOMPOSE,
    ),
    AnswerType.MIXED: (ProbeStrategy.COUNTER_EXAMPLE, ProbeStrategy.REQUEST_JUSTIFICATION),
    AnswerType.EVASIVE: (ProbeStrategy.DECOMPOSE, ProbeStrategy.SIMPLER_REASON),
    AnswerType.MANIPULATION: (ProbeStrategy.REFUSE_AND_REDIRECT,),
    AnswerType.UNSURE: (ProbeStrategy.REQUEST_JUSTIFICATION,),
}
