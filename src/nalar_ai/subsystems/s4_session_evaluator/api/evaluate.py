"""POST /v1/s4/sessions/evaluate and its DTOs (design doc §5)."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.shared.enums import ConceptOutcome, ProbeStrategy, RubricDimension
from nalar_ai.subsystems.s4_session_evaluator.api.dependencies import GatewayDep, S4Dep
from nalar_ai.subsystems.s4_session_evaluator.application.evaluate_session import (
    EvaluateSessionUseCase,
    EvaluationOutcome,
)
from nalar_ai.subsystems.s4_session_evaluator.application.write_reflection import (
    ReflectionSource,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.session import (
    RUBRIC_LEVELS,
    EvalMisconception,
    EvalPack,
    EvalSession,
    EvalTarget,
    EvalTurn,
    Rubric,
    TurnKind,
)

router = APIRouter()

Cue = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Term = Annotated[str, StringConstraints(min_length=1, max_length=100)]
Descriptor = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Levels = Annotated[list[Descriptor], Field(min_length=RUBRIC_LEVELS, max_length=RUBRIC_LEVELS)]


class EvalTargetIn(BaseModel):
    id: UUID
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)


class EvalMisconceptionIn(BaseModel):
    id: UUID
    concept_id: UUID
    statement: str = Field(min_length=1, max_length=1000)
    detection_cues: list[Cue] = Field(default_factory=list, max_length=20)


class EvaluationPackIn(BaseModel):
    """mission_versions.context_pack, sent unchanged. Only the fields S4 reads are declared;
    the rest (question bank, limits) is accepted and ignored."""

    model_config = ConfigDict(extra="ignore")

    pack_version: Literal[1]
    anchor_problem: str = Field(min_length=1, max_length=4000)
    reference_reasoning: str = Field(min_length=1, max_length=8000)
    targets: list[EvalTargetIn] = Field(min_length=1, max_length=5)
    misconceptions: list[EvalMisconceptionIn] = Field(default_factory=list, max_length=30)
    answer_terms: list[Term] = Field(min_length=1, max_length=50)

    def to_pack(self) -> EvalPack:
        return EvalPack(
            anchor_problem=self.anchor_problem,
            reference_reasoning=self.reference_reasoning,
            targets=tuple(EvalTarget(t.id, t.name, t.description) for t in self.targets),
            misconceptions=tuple(
                EvalMisconception(m.id, m.concept_id, m.statement, tuple(m.detection_cues))
                for m in self.misconceptions
            ),
            answer_terms=tuple(self.answer_terms),
        )


class RubricIn(BaseModel):
    """mission_versions.rubric: five descriptors per dimension; list position is the level."""

    claim: Levels
    evidence: Levels
    mechanism: Levels
    transfer: Levels

    def to_rubric(self) -> Rubric:
        return Rubric({d: tuple(getattr(self, d.value)) for d in RubricDimension})


class EvalTurnIn(BaseModel):
    """One session_turns row. S3's analysis of the answer is deliberately not accepted."""

    turn_id: UUID = Field(description="session_turns.id, returned on every quote")
    turn_index: int = Field(ge=0, le=50)
    kind: Literal["anchor", "probe"]
    question_text: str = Field(min_length=1, max_length=2000)
    answer_text: str | None = Field(
        default=None,
        max_length=20_000,
        description="session_turns.answer_text; null or empty when the turn was not answered",
    )
    move: ProbeStrategy | None = Field(default=None, description="prompt_strategy")
    target_concept_id: UUID | None = None

    def to_turn(self) -> EvalTurn:
        return EvalTurn(
            turn_id=self.turn_id,
            turn_index=self.turn_index,
            kind=TurnKind(self.kind),
            question_text=self.question_text,
            answer_text=self.answer_text or "",
            move=self.move,
            target_concept_id=self.target_concept_id,
        )


class EvaluateIn(BaseModel):
    context_pack: EvaluationPackIn
    rubric: RubricIn
    turns: list[EvalTurnIn] = Field(min_length=1, max_length=30)


class ScoreEvidenceOut(BaseModel):
    turn_id: UUID
    quote: str = Field(description="The student's own words, verbatim from that turn's answer")


class RubricScoreOut(BaseModel):
    dimension: RubricDimension
    level: int = Field(description="evaluation_scores.ai_level and final_level")
    rationale: str
    evidence: list[ScoreEvidenceOut] = Field(description="score_evidence rows; at least one")


class ConceptResultOut(BaseModel):
    concept_id: UUID
    outcome: ConceptOutcome
    misconception_id: UUID | None
    evidence_turn_id: UUID | None
    initial_misconception_id: UUID | None
    resolved_in_session: bool


class TurnQualityOut(BaseModel):
    turn_id: UUID
    turn_index: int
    quality: int = Field(description="0 empty or evasive … 4 correct with a clear mechanism")


class SessionReflectionOut(BaseModel):
    content: str = Field(description="session_reflections.content")
    source: ReflectionSource = Field(description="Not stored; 'template' after two blocked tries")


class EvaluateOut(BaseModel):
    summary: str = Field(description="session_evaluations.summary")
    scores: list[RubricScoreOut]
    concept_results: list[ConceptResultOut]
    turn_quality: list[TurnQualityOut] = Field(description="session_evaluations.turn_quality")
    reflection: SessionReflectionOut

    @classmethod
    def from_outcome(cls, outcome: EvaluationOutcome) -> "EvaluateOut":
        e = outcome.evaluation
        return cls(
            summary=e.summary,
            scores=[
                RubricScoreOut(
                    dimension=s.dimension,
                    level=s.level,
                    rationale=s.rationale,
                    evidence=[
                        ScoreEvidenceOut(turn_id=x.turn_id, quote=x.quote) for x in s.evidence
                    ],
                )
                for s in e.scores
            ],
            concept_results=[
                ConceptResultOut(
                    concept_id=c.concept_id,
                    outcome=c.outcome,
                    misconception_id=c.misconception_id,
                    evidence_turn_id=c.evidence_turn_id,
                    initial_misconception_id=c.initial_misconception_id,
                    resolved_in_session=c.resolved_in_session,
                )
                for c in e.concept_results
            ],
            turn_quality=[
                TurnQualityOut(turn_id=q.turn_id, turn_index=q.turn_index, quality=q.quality)
                for q in e.turn_quality
            ],
            reflection=SessionReflectionOut(
                content=outcome.reflection.parts.content, source=outcome.reflection.source
            ),
        )


@router.post("/sessions/evaluate", response_model=Envelope[EvaluateOut])
async def evaluate_session(
    body: EvaluateIn, ledger: LedgerDep, llm: GatewayDep, s4: S4Dep
) -> Envelope[EvaluateOut]:
    """Score a finished session and write the student's reflection (TC-10, ST-6).

    Call it once per session (NFR-R2). Every quote is verified to be the student's own words;
    an answer that cannot be verified fails with `ai_output_invalid` rather than being stored.
    """
    use_case = EvaluateSessionUseCase(llm=llm, config=s4.config, policy=s4.policy)
    outcome = await use_case.execute(
        EvalSession(
            pack=body.context_pack.to_pack(),
            rubric=body.rubric.to_rubric(),
            turns=tuple(turn.to_turn() for turn in body.turns),
        ),
        ledger,
    )
    return envelope(EvaluateOut.from_outcome(outcome), ledger, outcome.warnings)
