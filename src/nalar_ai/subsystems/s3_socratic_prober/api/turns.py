from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.shared.enums import (
    GuardResult,
    MoveReasonCode,
    MoveSource,
    PlannerMode,
    ProbeStrategy,
)
from nalar_ai.subsystems.s3_socratic_prober.api.dependencies import (
    EmbeddingsDep,
    GatewayDep,
    S3Dep,
)
from nalar_ai.subsystems.s3_socratic_prober.api.pack import ContextPackIn
from nalar_ai.subsystems.s3_socratic_prober.application.next_turn import (
    NextTurnCommand,
    NextTurnUseCase,
    QuestionSource,
    TurnAction,
    TurnResult,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import ClassificationSource
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import EndReason
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.session import HistoryTurn, TurnKind

router = APIRouter()


class HistoryTurnIn(BaseModel):
    """One stored session_turns row with its prober trace. The last turn holds the new answer."""

    turn_index: int = Field(ge=0, le=50)
    kind: TurnKind
    question_text: str = Field(min_length=1, max_length=2000)
    answer_text: str = Field(max_length=20_000)
    move: ProbeStrategy | None = Field(default=None, description="prompt_strategy")
    target_concept_id: UUID | None = None
    question_bank_id: str | None = Field(default=None, max_length=100)
    answer_type: AnswerType | None = Field(
        default=None, description="The stored analysis; null for the latest answer"
    )
    misconception_ids: list[UUID] = Field(
        default_factory=list,
        max_length=2,
        description="detected_misconception_id, then secondary_misconception_id",
    )

    def to_turn(self) -> HistoryTurn:
        return HistoryTurn(
            turn_index=self.turn_index,
            kind=self.kind,
            question_text=self.question_text,
            answer_text=self.answer_text,
            move=self.move,
            target_concept_id=self.target_concept_id,
            question_bank_id=self.question_bank_id,
            answer_type=self.answer_type,
            misconception_ids=tuple(self.misconception_ids),
        )


class NextTurnIn(BaseModel):
    planner_mode: PlannerMode = Field(description="publication_runs.planner_mode")
    context_pack: ContextPackIn
    history: list[HistoryTurnIn] = Field(min_length=1, max_length=30)
    elapsed_seconds: int = Field(ge=0, le=86_400)


class AnalysisOut(BaseModel):
    """The latest answer: answer_state, misconceptions and frustration for its turn row."""

    answer_type: AnswerType
    source: ClassificationSource
    misconception_id: UUID | None
    secondary_misconception_id: UUID | None
    frustration: bool
    key_phrase: str | None


class ProbeOut(BaseModel):
    """The next session_turns row."""

    move: ProbeStrategy
    allowed_moves: list[ProbeStrategy]
    target_concept_id: UUID
    question_bank_id: str
    question_text: str
    question_source: QuestionSource
    move_source: MoveSource
    reason_code: MoveReasonCode
    reason: str
    guard_result: GuardResult


class CoverageOut(BaseModel):
    concept_id: UUID
    challenged: bool


class NextTurnOut(BaseModel):
    action: TurnAction
    analysis: AnalysisOut
    probe: ProbeOut | None
    end_reason: EndReason | None
    safety_message: str | None
    coverage: list[CoverageOut]

    @classmethod
    def from_result(cls, result: TurnResult) -> "NextTurnOut":
        c = result.classification
        p = result.probe
        return cls(
            action=result.action,
            analysis=AnalysisOut(
                answer_type=c.answer_type,
                source=c.source,
                misconception_id=c.primary_misconception,
                secondary_misconception_id=c.secondary_misconception,
                frustration=c.frustration,
                key_phrase=c.key_phrase,
            ),
            probe=(
                ProbeOut(
                    move=p.move,
                    allowed_moves=list(p.allowed_moves),
                    target_concept_id=p.target_concept_id,
                    question_bank_id=p.question_bank_id,
                    question_text=p.question_text,
                    question_source=p.question_source,
                    move_source=p.move_source,
                    reason_code=p.reason_code,
                    reason=p.reason,
                    guard_result=p.guard_result,
                )
                if p is not None
                else None
            ),
            end_reason=result.end_reason,
            safety_message=result.safety_message,
            coverage=[
                CoverageOut(concept_id=concept, challenged=challenged)
                for concept, challenged in result.coverage
            ],
        )


@router.post("/turns/next", response_model=Envelope[NextTurnOut])
async def next_turn(
    body: NextTurnIn,
    ledger: LedgerDep,
    llm: GatewayDep,
    embeddings: EmbeddingsDep,
    s3: S3Dep,
) -> Envelope[NextTurnOut]:
    """Decide what follows the student's latest answer (ST-4): a probe, the end, or a safety pause.

    A model failure never fails the turn: it degrades to a teacher-approved question shown
    verbatim and is reported in `warnings`.
    """
    use_case = NextTurnUseCase(llm=llm, embeddings=embeddings, config=s3.config, policy=s3.policy)
    result = await use_case.execute(
        NextTurnCommand(
            pack=body.context_pack.to_pack(),
            history=tuple(turn.to_turn() for turn in body.history),
            mode=body.planner_mode,
            elapsed_seconds=body.elapsed_seconds,
        ),
        ledger,
    )
    return envelope(NextTurnOut.from_result(result), ledger, result.warnings)
