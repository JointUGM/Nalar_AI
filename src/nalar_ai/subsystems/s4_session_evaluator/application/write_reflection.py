"""Steps 3-4: Haiku writes the reflection, code guards it (design doc §8, §9).

The writer sees the student's own words only: no reference reasoning, no concept names, no
wrong-idea statements, no levels and no answer terms (information minimisation, as S3).
"""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s4_session_evaluator.application.policy import EvaluatorPolicy
from nalar_ai.subsystems.s4_session_evaluator.application.rendering import transcript
from nalar_ai.subsystems.s4_session_evaluator.domain.config import EvaluatorConfig
from nalar_ai.subsystems.s4_session_evaluator.domain.reflection import (
    ReflectionParts,
    check_reflection,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.session import EvalSession
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import Evaluation

TRIES = 2  # the first call and one fresh retry, then the template
STRENGTH_SCORES = 2  # quotes from the two highest-scoring dimensions

ReflectionSource = Literal["model", "template"]


class ReflectionOut(BaseModel):
    strengths: str
    changed_mind: str
    question: str


@dataclass(frozen=True, slots=True)
class ReflectionResult:
    parts: ReflectionParts
    source: ReflectionSource
    retried: bool


class WriteReflectionUseCase:
    PROMPT_ID = "s4.write_reflection"

    def __init__(
        self, *, llm: LLMGateway, config: EvaluatorConfig, policy: EvaluatorPolicy
    ) -> None:
        self._llm = llm
        self._config = config
        self._policy = policy

    async def execute(
        self, session: EvalSession, evaluation: Evaluation, ledger: UsageLedger
    ) -> ReflectionResult:
        """Provider errors propagate (the backend retries the whole evaluation, design doc §10);
        content problems end in the template."""
        answers = [turn.answer_text for turn in session.answered_turns]
        variables = self._variables(session, evaluation)
        feedback = "(none)"
        for attempt in range(TRIES):
            try:
                out = await self._llm.generate(
                    prompt_id=self.PROMPT_ID,
                    variables={**variables, "feedback": feedback},
                    output_model=ReflectionOut,
                    ledger=ledger,
                    policy=self._policy.reflect_call(),
                )
            except OutputValidationError:
                feedback = "- the answer was not valid JSON with the three parts"
                continue
            parts = ReflectionParts(out.strengths, out.changed_mind, out.question)
            problems = check_reflection(
                parts, answers, session.pack.answer_terms, self._config.lexicon
            )
            if not problems:
                return ReflectionResult(parts, "model", retried=attempt > 0)
            feedback = "\n".join(f"- {problem}" for problem in problems)
        return ReflectionResult(self._config.template, "template", retried=True)

    def _variables(self, session: EvalSession, evaluation: Evaluation) -> dict[str, str]:
        by_id = {turn.turn_id: turn for turn in session.turns}
        changed = [
            by_id[result.evidence_turn_id].alias
            for result in evaluation.concept_results
            if result.resolved_in_session and result.evidence_turn_id in by_id
        ]
        strongest = sorted(evaluation.scores, key=lambda score: -score.level)[:STRENGTH_SCORES]
        quotes = [
            f"- {by_id[e.turn_id].alias}: {fence_untrusted('student_quote', e.quote)}"
            for score in strongest
            for e in score.evidence
            if e.turn_id in by_id
        ]
        # Concept aliases in the transcript are opaque here: the writer never sees the names.
        opaque: AliasMap[UUID] = AliasMap("c", [t.id for t in session.pack.targets])
        return {
            "anchor": fence_untrusted("anchor", session.pack.anchor_problem),
            "transcript": transcript(
                session.turns, opaque, max_answer_chars=self._policy.max_answer_chars
            ),
            "changed": ", ".join(dict.fromkeys(changed)) or "(none)",
            "strengths": "\n".join(quotes) or "(none)",
        }
