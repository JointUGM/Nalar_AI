"""POST /v1/s5/teaching-plan (TC-23): one next-lesson activity for one held misconception."""

from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.subsystems.s5_insight_synthesizer.api.dependencies import GatewayDep, S5Dep
from nalar_ai.subsystems.s5_insight_synthesizer.application.teaching_plan import (
    PlanContext,
    TeachingPlanUseCase,
)

router = APIRouter()

Text = Field(min_length=1, max_length=1000)


class PlanQuestionIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Text


class TeachingPlanIn(BaseModel):
    """Reviewed content only: no student names, ids or counts (AI-4, invariant 8)."""

    mission_title: str = Field(min_length=1, max_length=300)
    learning_objective: str = Field(min_length=1, max_length=2000)
    concept_name: str = Field(min_length=1, max_length=300)
    misconception_id: UUID
    statement: str = Text
    correct_understanding: str = Field(min_length=1, max_length=2000)
    counter_examples: list[str] = Field(default_factory=list, max_length=5)
    detection_cues: list[str] = Field(default_factory=list, max_length=5)
    questions: list[PlanQuestionIn] = Field(
        default_factory=list, max_length=8, description="Approved bank questions on this concept"
    )


class PlanStepOut(BaseModel):
    phase: str = Field(description="elicit, contrast, explain")
    teacher_action: str
    ask: str


class TeachingPlanOut(BaseModel):
    misconception_id: UUID
    objective: str
    steps: list[PlanStepOut]
    counter_example: str | None = Field(
        description="The reviewed counter-example the contrast step uses, copied verbatim"
    )
    check_question: str
    look_for: str = Field(description="Teacher-only: the reasoning that shows change")
    question_ids: list[str]


@router.post("/teaching-plan", response_model=Envelope[TeachingPlanOut])
async def teaching_plan(
    body: TeachingPlanIn, ledger: LedgerDep, llm: GatewayDep, s5: S5Dep
) -> Envelope[TeachingPlanOut]:
    """Draft a short elicit, contrast, explain and check activity for the next lesson (TC-23).

    A teacher-reviewed draft, not a verdict. Fails with `ai_output_invalid` after two drafts
    fail the guard; show the reviewed strategy card instead.
    """
    ctx = PlanContext(
        mission_title=body.mission_title,
        learning_objective=body.learning_objective,
        concept_name=body.concept_name,
        statement=body.statement,
        correct_understanding=body.correct_understanding,
        counter_examples=tuple(e for e in body.counter_examples if e.strip()),
        detection_cues=tuple(c for c in body.detection_cues if c.strip()),
        questions=tuple((q.id, q.text) for q in body.questions),
    )
    use_case = TeachingPlanUseCase(llm=llm, config=s5.config, policy=s5.policy)
    result = await use_case.execute(ctx, ledger)
    plan = result.plan
    out = TeachingPlanOut(
        misconception_id=body.misconception_id,
        objective=plan.objective,
        steps=[
            PlanStepOut(phase=s.phase, teacher_action=s.teacher_action, ask=s.ask)
            for s in plan.steps
        ],
        counter_example=ctx.counter_examples[plan.counter_example - 1]
        if plan.counter_example is not None
        else None,
        check_question=plan.check_question,
        look_for=plan.look_for,
        question_ids=list(plan.question_ids),
    )
    return envelope(out, ledger, ["plan_retried"] if result.retried else [])
