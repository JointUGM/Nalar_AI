"""Step 5: Sonnet picks one allowed move and question and adapts it (design doc §9).

Built only from a WriterView: the reference reasoning, concept names and answer terms can't
reach this prompt, because nothing here can hold them (AI-1).
"""

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import CallPolicy, LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import MoveReasonCode, ProbeStrategy
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s3_socratic_prober.application.classify_answer import (
    LIVE_MODEL_FAILURES,
)
from nalar_ai.subsystems.s3_socratic_prober.application.rendering import (
    NONE,
    bank_block,
    transcript,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import Classification
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import MovePlan
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import WriterView
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView
from nalar_ai.subsystems.s3_socratic_prober.domain.validation import ModelChoice


class ChooseOut(BaseModel):
    move: ProbeStrategy
    question_id: str
    reason_code: MoveReasonCode
    reason: str
    question: str


class ChoiceFailure(StrEnum):
    ERROR = "error"  # failed, slow, or no time left → fallback_error
    INVALID = "invalid"  # unparseable output → fallback_invalid


class _Aliases:
    def __init__(self, writer: WriterView) -> None:
        self.concepts = AliasMap("c", writer.target_ids)
        self.misconceptions = AliasMap("m", writer.misconception_ids)
        self.questions = AliasMap("q", [q.id for q in writer.question_bank])


class ChooseProbeUseCase:
    PROMPT_ID = "s3.choose_probe"

    def __init__(self, *, llm: LLMGateway, transcript_turns: int, max_answer_chars: int) -> None:
        self._llm = llm
        self._transcript_turns = transcript_turns
        self._max_answer_chars = max_answer_chars

    async def execute(
        self,
        writer: WriterView,
        view: SessionView,
        classification: Classification,
        plan: MovePlan,
        ledger: UsageLedger,
        *,
        timeout_s: float | None,
    ) -> ModelChoice | ChoiceFailure:
        if timeout_s is None:
            return ChoiceFailure.ERROR
        aliases = _Aliases(writer)
        try:
            out = await self._llm.generate(
                prompt_id=self.PROMPT_ID,
                variables=self._variables(writer, view, classification, plan, aliases),
                output_model=ChooseOut,
                ledger=ledger,
                policy=CallPolicy(timeout_s=timeout_s, max_attempts=1, max_repairs=0, lane="live"),
            )
        except OutputValidationError:
            return ChoiceFailure.INVALID
        except LIVE_MODEL_FAILURES:
            return ChoiceFailure.ERROR
        question_id = (
            aliases.questions.resolve(out.question_id)
            if out.question_id in aliases.questions
            else None
        )
        return ModelChoice(
            move=out.move,
            question_id=question_id,
            reason_code=out.reason_code,
            reason=out.reason,
            question=out.question,
        )

    def _variables(
        self,
        writer: WriterView,
        view: SessionView,
        classification: Classification,
        plan: MovePlan,
        aliases: _Aliases,
    ) -> dict[str, str]:
        return {
            "bank": bank_block(
                writer.target_ids,
                writer.question_bank,
                aliases.concepts,
                aliases.questions,
                aliases.misconceptions,
            ),
            "target": aliases.concepts.alias(plan.target_concept_id),
            "answer_type": classification.answer_type.value,
            "detected": _listed(
                aliases.misconceptions.alias(m) for m in classification.misconception_ids
            ),
            "signals": _listed(
                signal
                for signal, active in (
                    ("the student sounds frustrated", classification.frustration),
                    ("coverage rule: this must be a challenge question", plan.coverage_forced),
                )
                if active
            ),
            "allowed": "\n".join(
                f"- {move.value}{' (default)' if move is plan.default_move else ''}: "
                + ", ".join(aliases.questions.alias(q.id) for q in plan.candidates[move])
                for move in plan.allowed_moves
            ),
            "transcript": transcript(
                view.turns[-self._transcript_turns :], max_answer_chars=self._max_answer_chars
            ),
            "key_phrase": (
                fence_untrusted("key_phrase", classification.key_phrase)
                if classification.key_phrase
                else NONE
            ),
        }


def _listed(items: Iterable[str]) -> str:
    values = list(items)
    return ", ".join(values) if values else NONE
