"""What S4 evaluates: the mission's pack and rubric plus the session's turns (design doc §5.1)."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from nalar_ai.shared.enums import ProbeStrategy, RubricDimension
from nalar_ai.shared.text import normalize_key

RUBRIC_LEVELS = 5  # levels 0..4


@dataclass(frozen=True, slots=True)
class EvalTarget:
    id: UUID
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class EvalMisconception:
    id: UUID
    concept_id: UUID
    statement: str
    detection_cues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalPack:
    """The part of mission_versions.context_pack that S4 reads. The question bank is ignored."""

    anchor_problem: str
    reference_reasoning: str
    targets: tuple[EvalTarget, ...]
    misconceptions: tuple[EvalMisconception, ...]
    answer_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rubric:
    """mission_versions.rubric: five descriptors per dimension; list position is the level."""

    descriptors: Mapping[RubricDimension, tuple[str, ...]]


class TurnKind(StrEnum):
    ANCHOR = "anchor"
    PROBE = "probe"


@dataclass(frozen=True, slots=True)
class EvalTurn:
    """One session_turns row, without S3's analysis of the answer (fresh eyes, design doc C3)."""

    turn_id: UUID
    turn_index: int
    kind: TurnKind
    question_text: str
    answer_text: str
    move: ProbeStrategy | None = None
    target_concept_id: UUID | None = None

    @property
    def answered(self) -> bool:
        return bool(self.answer_text.strip())

    @property
    def alias(self) -> str:
        return f"t{self.turn_index}"


@dataclass(frozen=True, slots=True)
class EvalSession:
    pack: EvalPack
    rubric: Rubric
    turns: tuple[EvalTurn, ...]

    @property
    def answered_turns(self) -> tuple[EvalTurn, ...]:
        return tuple(turn for turn in self.turns if turn.answered)


def validate_session(session: EvalSession) -> list[str]:
    """Structural problems, each a 422 (design doc §5.1). Empty means valid."""
    errors: list[str] = []
    pack = session.pack
    target_ids = [target.id for target in pack.targets]
    if len(set(target_ids)) != len(target_ids):
        errors.append("context_pack.targets: duplicate id")
    misconception_ids = [m.id for m in pack.misconceptions]
    if len(set(misconception_ids)) != len(misconception_ids):
        errors.append("context_pack.misconceptions: duplicate id")
    for m in pack.misconceptions:
        if m.concept_id not in target_ids:
            errors.append(f"misconception {m.id}: concept {m.concept_id} is not a target")
    for dimension in RubricDimension:
        if len(session.rubric.descriptors.get(dimension, ())) != RUBRIC_LEVELS:
            errors.append(f"rubric.{dimension.value}: needs exactly {RUBRIC_LEVELS} descriptors")
    turns = session.turns
    if not turns:
        return [*errors, "turns: at least the anchor turn is required"]
    if len({turn.turn_id for turn in turns}) != len(turns):
        errors.append("turns: duplicate turn_id")
    if [turn.turn_index for turn in turns] != list(range(len(turns))):
        errors.append("turns: turn_index must run 0, 1, 2, ... in order")
    for turn in turns:
        if (turn.turn_index == 0) != (turn.kind is TurnKind.ANCHOR):
            errors.append(f"turn {turn.turn_index}: only turn 0 is the anchor")
        if turn.kind is TurnKind.PROBE and (turn.move is None or turn.target_concept_id is None):
            errors.append(f"turn {turn.turn_index}: a probe needs a move and a target")
        if turn.target_concept_id is not None and turn.target_concept_id not in target_ids:
            errors.append(f"turn {turn.turn_index}: target is not in the context pack")
    if not session.answered_turns:
        errors.append("turns: no answered turn to evaluate")
    elif not any(normalize_key(turn.answer_text) for turn in session.answered_turns):
        # Every score must quote the student's words (design doc S4-D2); "..." has none.
        errors.append("turns: no answer contains words to quote")
    return errors
