"""Steps 3-4 in code: end rules, target concept, allowed moves and hard rules (design doc §8)."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from nalar_ai.shared.enums import PlannerMode, ProbeStrategy
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import Classification
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import (
    MISCONCEPTION_TYPES,
    MOVE_TABLE,
    AnswerType,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import BankQuestion, ContextPack
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView

MAX_UNCHANGED_COUNTERS = 2
MAX_DECOMPOSES_IN_A_ROW = 2


class EndReason(StrEnum):
    COVERAGE_COMPLETE = "coverage_complete"
    TURN_LIMIT = "turn_limit"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, slots=True)
class MovePlan:
    """What the chooser may do this turn. Every allowed move has at least one candidate."""

    target_concept_id: UUID
    allowed_moves: tuple[ProbeStrategy, ...]  # default first
    candidates: Mapping[ProbeStrategy, tuple[BankQuestion, ...]]
    coverage_forced: bool

    @property
    def default_move(self) -> ProbeStrategy:
        return self.allowed_moves[0]

    @property
    def default_question(self) -> BankQuestion:
        return self.candidates[self.default_move][0]

    def candidate(self, move: ProbeStrategy, question_id: str) -> BankQuestion | None:
        return next((q for q in self.candidates.get(move, ()) if q.id == question_id), None)


def check_end(
    view: SessionView, pack: ContextPack, *, min_probes: int, elapsed_seconds: int
) -> EndReason | None:
    if elapsed_seconds >= pack.max_duration_minutes * 60:
        return EndReason.TIME_LIMIT
    if view.probes_asked >= pack.max_probes:
        return EndReason.TURN_LIMIT
    all_challenged = set(pack.target_ids) <= view.challenged
    if all_challenged and view.probes_asked >= min(min_probes, pack.max_probes):
        return EndReason.COVERAGE_COMPLETE
    return None


def plan_move(
    classification: Classification, view: SessionView, pack: ContextPack, *, mode: PlannerMode
) -> MovePlan:
    answer_type = classification.answer_type
    if answer_type is AnswerType.SAFETY:
        raise ValueError("a safety answer pauses the session; it has no move")
    unchallenged = [t for t in pack.target_ids if t not in view.challenged]
    remaining = pack.max_probes - view.probes_asked
    forced = bool(unchallenged) and remaining <= len(unchallenged)
    primary = classification.primary_misconception
    misconception = pack.misconception(primary) if primary is not None else None
    misconception_concept = misconception.concept_id if misconception else None
    target = _pick_target(view, pack, unchallenged, misconception_concept, forced)

    moves: list[ProbeStrategy]
    if forced:
        counters_first = answer_type in MISCONCEPTION_TYPES and misconception_concept == target
        moves = (
            [ProbeStrategy.COUNTER_EXAMPLE, ProbeStrategy.TRANSFER]
            if counters_first
            else [ProbeStrategy.TRANSFER, ProbeStrategy.COUNTER_EXAMPLE]
        )
    else:
        moves = list(MOVE_TABLE[answer_type])
        if (
            primary is not None
            and view.counter_streak(primary, classification.misconception_ids)
            >= MAX_UNCHANGED_COUNTERS
        ):
            moves = [m for m in moves if m is not ProbeStrategy.COUNTER_EXAMPLE]
        if view.trailing_decomposes >= MAX_DECOMPOSES_IN_A_ROW:
            moves = [m for m in moves if m is not ProbeStrategy.DECOMPOSE]
        if not moves:
            moves = [ProbeStrategy.REQUEST_JUSTIFICATION]
    if mode is PlannerMode.TABLE:
        moves = moves[:1]

    candidates = {
        move: _candidates(pack, target, move, primary, view.asked_question_ids) for move in moves
    }
    allowed = tuple(move for move in moves if candidates[move])
    if not allowed:
        raise ValueError(f"no approved question for target {target}; validate the pack first")
    return MovePlan(
        target_concept_id=target,
        allowed_moves=allowed,
        candidates={move: candidates[move] for move in allowed},
        coverage_forced=forced,
    )


def coverage(pack: ContextPack, challenged: frozenset[UUID]) -> tuple[tuple[UUID, bool], ...]:
    return tuple((target, target in challenged) for target in pack.target_ids)


def _pick_target(
    view: SessionView,
    pack: ContextPack,
    unchallenged: list[UUID],
    misconception_concept: UUID | None,
    forced: bool,
) -> UUID:
    if forced:
        for preferred in (misconception_concept, view.last_target):
            if preferred is not None and preferred in unchallenged:
                return preferred
        return unchallenged[0]
    if misconception_concept is not None:
        return misconception_concept
    if view.last_target is None:
        return pack.target_ids[0]
    if view.last_target in view.challenged and unchallenged:
        return unchallenged[0]
    return view.last_target


def _candidates(
    pack: ContextPack,
    target: UUID,
    move: ProbeStrategy,
    misconception_id: UUID | None,
    asked: frozenset[str],
) -> tuple[BankQuestion, ...]:
    questions = [q for q in pack.question_bank if q.concept_id == target and q.move is move]
    if move is ProbeStrategy.COUNTER_EXAMPLE and misconception_id is not None:
        matching = [q for q in questions if q.misconception_id == misconception_id]
        questions = matching + [q for q in questions if q.misconception_id != misconception_id]
    fresh = [q for q in questions if q.id not in asked]
    return tuple(fresh or questions)
