"""TC-23: draft one next-lesson activity from reviewed content; code guards its shape."""

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s5_insight_synthesizer.application.policy import SynthesizerPolicy
from nalar_ai.subsystems.s5_insight_synthesizer.domain.config import SynthesizerConfig
from nalar_ai.subsystems.s5_insight_synthesizer.domain.teaching_plan import (
    PlanStep,
    TeachingPlan,
    check_plan,
)

TRIES = 2  # the first call and one fresh retry, then 502

logger = logging.getLogger("nalar_ai.s5")


@dataclass(frozen=True, slots=True)
class PlanContext:
    """Reviewed teaching content for one misconception. Every string is teacher text."""

    mission_title: str
    learning_objective: str
    concept_name: str
    statement: str
    correct_understanding: str
    counter_examples: tuple[str, ...]
    detection_cues: tuple[str, ...]
    questions: tuple[tuple[str, str], ...]  # (id, text) from the reviewed question bank


class StepDraftOut(BaseModel):
    phase: str = Field(description="elicit, contrast or explain, in that order")
    teacher_action: str = Field(description="What the teacher does or says, Bahasa Indonesia")
    ask: str = Field(description="One question for students, without its answer")


class PlanDraftOut(BaseModel):
    objective: str
    steps: list[StepDraftOut] = Field(description="Exactly three steps")
    counter_example: int | None = Field(
        description="Number of the reviewed counter-example used, or null when none is listed"
    )
    check_question: str
    look_for: str
    question_ids: list[str] = Field(description="Approved question ids used; may be empty")


@dataclass(frozen=True, slots=True)
class PlanResult:
    plan: TeachingPlan
    retried: bool


def _numbered(items: tuple[str, ...]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1)) or "(none)"


class TeachingPlanUseCase:
    PROMPT_ID = "s5.teaching_plan"

    def __init__(
        self, *, llm: LLMGateway, config: SynthesizerConfig, policy: SynthesizerPolicy
    ) -> None:
        self._llm = llm
        self._config = config
        self._policy = policy

    async def execute(self, ctx: PlanContext, ledger: UsageLedger) -> PlanResult:
        """Provider errors propagate (503); two guarded failures raise ai_output_invalid."""
        variables = {
            "title": fence_untrusted("mission", ctx.mission_title),
            "objective": fence_untrusted("objective", ctx.learning_objective),
            "concept": fence_untrusted("concept", ctx.concept_name),
            "statement": fence_untrusted("idea", ctx.statement),
            "correct": fence_untrusted("correct", ctx.correct_understanding),
            "examples": fence_untrusted("examples", _numbered(ctx.counter_examples)),
            "cues": fence_untrusted("cues", _numbered(ctx.detection_cues)),
            "questions": fence_untrusted(
                "questions", "\n".join(f"{qid}: {text}" for qid, text in ctx.questions) or "(none)"
            ),
        }
        known_ids = frozenset(qid for qid, _ in ctx.questions)
        feedback, draft, problems = "(none)", "(none)", ["no attempt"]
        for attempt in range(TRIES):
            try:
                out = await self._llm.generate(
                    prompt_id=self.PROMPT_ID,
                    variables={**variables, "feedback": feedback, "draft": draft},
                    output_model=PlanDraftOut,
                    ledger=ledger,
                    policy=self._policy.plan_call(),
                )
            except OutputValidationError as error:
                # Pydantic paths and messages only ("steps: Field required"), never the input.
                problems = [
                    f"invalid JSON field {found}" for found in error.details.get("errors", [])
                ] or ["the answer must be valid JSON with every required field"]
            else:
                plan = TeachingPlan(
                    objective=out.objective.strip(),
                    steps=tuple(
                        PlanStep(s.phase.strip(), s.teacher_action.strip(), s.ask.strip())
                        for s in out.steps
                    ),
                    counter_example=out.counter_example,
                    check_question=out.check_question.strip(),
                    look_for=out.look_for.strip(),
                    question_ids=tuple(dict.fromkeys(out.question_ids)),
                )
                problems = check_plan(
                    plan,
                    counter_examples=len(ctx.counter_examples),
                    question_ids=known_ids,
                    lexicon=self._config.numbers,
                    limits=self._config.teaching_plan,
                )
                if not problems:
                    return PlanResult(plan, retried=attempt > 0)
                draft = fence_untrusted("previous_draft", out.model_dump_json())
            # Problems name fields and lexicon terms only, never teacher text.
            logger.warning(
                "s5 guard prompt=%s request_id=%s checks=%s",
                self.PROMPT_ID,
                ledger.request_id,
                "; ".join(problems[:10]),
            )
            feedback = "\n".join(f"- {problem}" for problem in problems)
        raise OutputValidationError(
            f"{self.PROMPT_ID} failed its guard {TRIES} times",
            details={"errors": problems[:10]},
        )
