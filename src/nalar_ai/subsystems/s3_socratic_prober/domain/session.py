"""The turn history the backend sends every turn (design doc §3, §14), and what code derives from it."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from nalar_ai.shared.enums import ProbeStrategy
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import CHALLENGE_MOVES, AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack


class TurnKind(StrEnum):
    ANCHOR = "anchor"
    PROBE = "probe"


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """One stored session_turns row. The last turn holds the answer being decided on."""

    turn_index: int
    kind: TurnKind
    question_text: str
    answer_text: str
    move: ProbeStrategy | None = None
    target_concept_id: UUID | None = None
    question_bank_id: str | None = None
    answer_type: AnswerType | None = None  # None: not analysed yet, or unknown
    misconception_ids: tuple[UUID, ...] = ()


def validate_history(history: Sequence[HistoryTurn], pack: ContextPack) -> list[str]:
    """Structural problems only. Analysis fields are read leniently (design doc §14)."""
    if not history:
        return ["history is empty; it must start with the anchor turn"]
    problems: list[str] = []
    if [turn.turn_index for turn in history] != list(range(len(history))):
        problems.append("turn_index values must be 0, 1, 2, ... in order")
    targets = set(pack.target_ids)
    for position, turn in enumerate(history):
        expected = TurnKind.ANCHOR if position == 0 else TurnKind.PROBE
        if turn.kind is not expected:
            problems.append(f"turn {position} must have kind {expected.value!r}")
        if turn.kind is TurnKind.PROBE:
            if turn.move is None:
                problems.append(f"probe turn {position} has no move")
            if turn.target_concept_id not in targets:
                problems.append(f"probe turn {position} targets a concept outside the pack")
    return problems


@dataclass(frozen=True, slots=True)
class SessionView:
    """Read-only facts the controller needs, derived from the history."""

    turns: tuple[HistoryTurn, ...]

    @property
    def latest(self) -> HistoryTurn:
        return self.turns[-1]

    @property
    def probes(self) -> tuple[HistoryTurn, ...]:
        return tuple(turn for turn in self.turns if turn.kind is TurnKind.PROBE)

    @property
    def probes_asked(self) -> int:
        return len(self.probes)

    @property
    def challenged(self) -> frozenset[UUID]:
        return frozenset(
            turn.target_concept_id
            for turn in self.probes
            if turn.move in CHALLENGE_MOVES and turn.target_concept_id is not None
        )

    @property
    def last_target(self) -> UUID | None:
        probes = self.probes
        return probes[-1].target_concept_id if probes else None

    @property
    def asked_question_ids(self) -> frozenset[str]:
        return frozenset(t.question_bank_id for t in self.probes if t.question_bank_id)

    @property
    def trailing_decomposes(self) -> int:
        count = 0
        for turn in reversed(self.probes):
            if turn.move is not ProbeStrategy.DECOMPOSE:
                break
            count += 1
        return count

    def counter_streak(self, misconception_id: UUID, latest_ids: Sequence[UUID]) -> int:
        """Counter-examples in a row that left `misconception_id` unchanged.

        A probe turn counters the misconception of the answer before it; it is "unchanged"
        when the probe's own answer still shows it. `latest_ids` is the fresh analysis of the
        latest answer, which the history does not hold yet.
        """
        streak = 0
        for position in range(len(self.turns) - 1, 0, -1):
            turn = self.turns[position]
            if turn.move is not ProbeStrategy.COUNTER_EXAMPLE:
                break
            before = self.turns[position - 1].misconception_ids
            after = latest_ids if position == len(self.turns) - 1 else turn.misconception_ids
            if misconception_id not in before or misconception_id not in after:
                break
            streak += 1
        return streak

    def student_answers(self) -> tuple[str, ...]:
        return tuple(turn.answer_text for turn in self.turns)
