"""Step 6: code checks the chooser's pick before anything else happens (design doc §10)."""

from dataclasses import dataclass

from nalar_ai.shared.enums import MoveReasonCode, ProbeStrategy
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import MovePlan
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import BankQuestion


@dataclass(frozen=True, slots=True)
class ModelChoice:
    """The chooser's output. `question_id` is None when its alias was unknown."""

    move: ProbeStrategy
    question_id: str | None
    reason_code: MoveReasonCode
    reason: str
    question: str


@dataclass(frozen=True, slots=True)
class AcceptedChoice:
    move: ProbeStrategy
    question: BankQuestion
    reason_code: MoveReasonCode
    reason: str
    adapted_text: str


@dataclass(frozen=True, slots=True)
class Rejection:
    problem: str


def validate_choice(choice: ModelChoice, plan: MovePlan) -> AcceptedChoice | Rejection:
    if choice.move not in plan.allowed_moves:
        return Rejection(f"move {choice.move.value} is not allowed")
    question = (
        plan.candidate(choice.move, choice.question_id) if choice.question_id is not None else None
    )
    if question is None:
        return Rejection(f"the question is not a candidate of move {choice.move.value}")
    is_default = choice.move is plan.default_move
    if is_default and choice.reason_code is not MoveReasonCode.DEFAULT:
        return Rejection("the default move must use reason code default")
    if not is_default and choice.reason_code is MoveReasonCode.DEFAULT:
        return Rejection("leaving the default move needs a reason code")
    if not is_default and not choice.reason.strip():
        return Rejection("leaving the default move needs a reason")
    if not choice.question.strip():
        return Rejection("the adapted question is empty")
    return AcceptedChoice(
        move=choice.move,
        question=question,
        reason_code=choice.reason_code,
        reason=choice.reason.strip(),
        adapted_text=choice.question.strip(),
    )
