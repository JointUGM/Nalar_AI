"""The per-turn pipeline (design doc §4):
prefilter → classify → end rules → plan → choose → validate → guard.

Every failure degrades to a teacher-approved question shown verbatim; nothing here fails
the turn except an invalid request (422) or a bug.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.enums import (
    GuardResult,
    MoveReasonCode,
    MoveSource,
    PlannerMode,
    ProbeStrategy,
)
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.choose_probe import (
    ChoiceFailure,
    ChooseProbeUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.application.classify_answer import (
    ClassifyAnswerUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.application.deadline import Deadline
from nalar_ai.subsystems.s3_socratic_prober.application.guard_probe import GuardProbeUseCase
from nalar_ai.subsystems.s3_socratic_prober.application.policy import ProberPolicy
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.config import ProberConfig
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import (
    EndReason,
    MovePlan,
    check_end,
    coverage,
    plan_move,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import CHALLENGE_MOVES, AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import (
    BankQuestion,
    ContextPack,
    validate_pack,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.prefilter import PrefilterHit, prefilter
from nalar_ai.subsystems.s3_socratic_prober.domain.session import (
    HistoryTurn,
    SessionView,
    validate_history,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.validation import Rejection, validate_choice

logger = logging.getLogger("nalar_ai.s3")


class TurnAction(StrEnum):
    PROBE = "probe"
    END = "end"
    SAFETY_PAUSE = "safety_pause"


class QuestionSource(StrEnum):
    ADAPTED = "adapted"  # the chooser's wording, which passed the guard
    APPROVED = "approved"  # the teacher-approved text, verbatim


@dataclass(frozen=True, slots=True)
class ProbeDecision:
    """The next session_turns row, as the backend stores it (design doc §14)."""

    move: ProbeStrategy
    allowed_moves: tuple[ProbeStrategy, ...]
    target_concept_id: UUID
    question_bank_id: str
    question_text: str
    question_source: QuestionSource
    move_source: MoveSource
    reason_code: MoveReasonCode
    reason: str
    guard_result: GuardResult


@dataclass(frozen=True, slots=True)
class TurnResult:
    action: TurnAction
    classification: Classification  # of the latest answer
    probe: ProbeDecision | None
    end_reason: EndReason | None
    safety_message: str | None
    coverage: tuple[tuple[UUID, bool], ...]  # after this decision
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NextTurnCommand:
    pack: ContextPack
    history: tuple[HistoryTurn, ...]
    mode: PlannerMode
    elapsed_seconds: int


class NextTurnUseCase:
    def __init__(
        self,
        *,
        llm: LLMGateway,
        embeddings: EmbeddingService,
        config: ProberConfig,
        policy: ProberPolicy,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._config = config
        self._policy = policy
        self._clock = clock
        self._classify = ClassifyAnswerUseCase(
            llm=llm,
            transcript_turns=policy.transcript_turns,
            max_answer_chars=policy.max_answer_chars,
        )
        self._choose = ChooseProbeUseCase(
            llm=llm,
            transcript_turns=policy.transcript_turns,
            max_answer_chars=policy.max_answer_chars,
        )
        self._guard = GuardProbeUseCase(
            embeddings=embeddings, lexicon=config.guard, thresholds=policy.guard
        )

    async def execute(self, command: NextTurnCommand, ledger: UsageLedger) -> TurnResult:
        pack = command.pack
        problems = validate_pack(pack) + validate_history(command.history, pack)
        if problems:
            raise InvalidInputError(
                "the context pack or the history is invalid", details={"problems": problems}
            )
        deadline = Deadline(
            self._policy.turn_budget_s, min_step_s=self._policy.min_step_s, clock=self._clock
        )
        view = SessionView(command.history)
        warnings: list[str] = []

        hit = prefilter(view.latest.answer_text, self._config.prefilter)
        if hit is not None:
            classification = Classification.from_prefilter(hit)
        else:
            classification = await self._classify.execute(
                pack, view, ledger, timeout_s=deadline.step(self._policy.classify_timeout_s)
            )
            if classification.source is ClassificationSource.FALLBACK:
                warnings.append("classification_fallback")

        if hit is PrefilterHit.SAFETY or classification.answer_type is AnswerType.SAFETY:
            result = TurnResult(
                action=TurnAction.SAFETY_PAUSE,
                classification=classification,
                probe=None,
                end_reason=None,
                safety_message=self._config.texts.safety_message,
                coverage=coverage(pack, view.challenged),
                warnings=tuple(warnings),
            )
            return self._logged(result, ledger)

        end = check_end(
            view,
            pack,
            min_probes=self._policy.min_probes,
            elapsed_seconds=command.elapsed_seconds,
        )
        if end is not None:
            result = TurnResult(
                action=TurnAction.END,
                classification=classification,
                probe=None,
                end_reason=end,
                safety_message=None,
                coverage=coverage(pack, view.challenged),
                warnings=tuple(warnings),
            )
            return self._logged(result, ledger)

        plan = plan_move(classification, view, pack, mode=command.mode)
        probe = await self._probe(command, view, classification, plan, deadline, ledger, warnings)
        challenged = view.challenged | (
            {probe.target_concept_id} if probe.move in CHALLENGE_MOVES else set()
        )
        result = TurnResult(
            action=TurnAction.PROBE,
            classification=classification,
            probe=probe,
            end_reason=None,
            safety_message=None,
            coverage=coverage(pack, frozenset(challenged)),
            warnings=tuple(warnings),
        )
        return self._logged(result, ledger)

    async def _probe(
        self,
        command: NextTurnCommand,
        view: SessionView,
        classification: Classification,
        plan: MovePlan,
        deadline: Deadline,
        ledger: UsageLedger,
        warnings: list[str],
    ) -> ProbeDecision:
        if classification.source is ClassificationSource.PREFILTER:
            return self._verbatim(plan, classification, MoveSource.PREFILTER)
        choice = await self._choose.execute(
            command.pack.writer_view(),
            view,
            classification,
            plan,
            ledger,
            timeout_s=deadline.step(self._policy.choose_timeout_s),
        )
        if isinstance(choice, ChoiceFailure):
            warnings.append("choice_fallback")
            source = (
                MoveSource.FALLBACK_ERROR
                if choice is ChoiceFailure.ERROR
                else MoveSource.FALLBACK_INVALID
            )
            return self._verbatim(plan, classification, source)
        checked = validate_choice(choice, plan)
        if isinstance(checked, Rejection):
            warnings.append("choice_fallback")
            logger.info("s3 choice rejected request_id=%s: %s", ledger.request_id, checked.problem)
            return self._verbatim(plan, classification, MoveSource.FALLBACK_INVALID)

        source = self._move_source(command.mode, plan)
        reason = (
            checked.reason
            if checked.reason_code is not MoveReasonCode.DEFAULT
            else self._config.texts.reason(
                move=checked.move,
                answer_type=classification.answer_type,
                move_source=source,
                coverage_forced=plan.coverage_forced,
            )
        )
        approved = checked.question.text
        text, question_source, guard_result = approved, QuestionSource.APPROVED, GuardResult.NOT_RUN
        if checked.adapted_text != approved.strip():
            outcome = await self._guard.execute(
                adapted=checked.adapted_text,
                approved=approved,
                pack=command.pack,
                view=view,
                ledger=ledger,
                timeout_s=deadline.step(self._policy.embed_timeout_s),
            )
            if outcome.degraded:
                warnings.append("guard_unavailable")
            guard_result = outcome.result
            if outcome.passed:
                text, question_source = checked.adapted_text, QuestionSource.ADAPTED
        return ProbeDecision(
            move=checked.move,
            allowed_moves=plan.allowed_moves,
            target_concept_id=plan.target_concept_id,
            question_bank_id=checked.question.id,
            question_text=text,
            question_source=question_source,
            move_source=source,
            reason_code=checked.reason_code,
            reason=reason,
            guard_result=guard_result,
        )

    def _verbatim(
        self, plan: MovePlan, classification: Classification, source: MoveSource
    ) -> ProbeDecision:
        question: BankQuestion = plan.default_question
        return ProbeDecision(
            move=plan.default_move,
            allowed_moves=plan.allowed_moves,
            target_concept_id=plan.target_concept_id,
            question_bank_id=question.id,
            question_text=question.text,
            question_source=QuestionSource.APPROVED,
            move_source=source,
            reason_code=MoveReasonCode.DEFAULT,
            reason=self._config.texts.reason(
                move=plan.default_move,
                answer_type=classification.answer_type,
                move_source=source,
                coverage_forced=plan.coverage_forced,
            ),
            guard_result=GuardResult.NOT_RUN,
        )

    @staticmethod
    def _move_source(mode: PlannerMode, plan: MovePlan) -> MoveSource:
        if mode is PlannerMode.TABLE:
            return MoveSource.DEFAULT
        if len(plan.allowed_moves) == 1:
            return MoveSource.FIXED_RULE
        return MoveSource.PLANNER

    def _logged(self, result: TurnResult, ledger: UsageLedger) -> TurnResult:
        probe = result.probe
        logger.info(
            "s3 turn request_id=%s action=%s answer_type=%s source=%s move=%s move_source=%s "
            "guard=%s calls=%d cost_usd=%.6f lexicon_v=%d",
            ledger.request_id,
            result.action.value,
            result.classification.answer_type.value,
            result.classification.source.value,
            probe.move.value if probe else "-",
            probe.move_source.value if probe else "-",
            probe.guard_result.value if probe else "-",
            len(ledger.records),
            ledger.total_cost_usd,
            self._config.version,
        )
        return result
