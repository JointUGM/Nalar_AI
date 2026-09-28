"""Step 2: Haiku labels the latest answer (design doc §7). Never fails the turn."""

from typing import Literal

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import CallPolicy, LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.errors import (
    ModelCallRejectedError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s3_socratic_prober.application.rendering import (
    clip,
    concepts_block,
    misconceptions_block,
    transcript,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ModelLabels,
    normalize_labels,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView

# Model failures on the live path degrade to "unsure"; bugs (budget, config) still raise.
LIVE_MODEL_FAILURES = (UpstreamUnavailableError, ModelCallRejectedError, OutputValidationError)


class ClassifyOut(BaseModel):
    answer_type: Literal[
        "correct_reasoned",
        "correct_unreasoned",
        "misconception",
        "mixed",
        "evasive",
        "manipulation",
    ]
    confidence: Literal["high", "low"]
    misconceptions: list[str]
    frustration: bool
    safety_concern: bool
    key_phrase: str


class ClassifyAnswerUseCase:
    PROMPT_ID = "s3.classify_answer"

    def __init__(self, *, llm: LLMGateway, transcript_turns: int, max_answer_chars: int) -> None:
        self._llm = llm
        self._transcript_turns = transcript_turns
        self._max_answer_chars = max_answer_chars

    async def execute(
        self,
        pack: ContextPack,
        view: SessionView,
        ledger: UsageLedger,
        *,
        timeout_s: float | None,
    ) -> Classification:
        """timeout_s None means the turn budget is spent: skip the call."""
        if timeout_s is None:
            return Classification.fallback()
        latest = view.latest
        concepts = AliasMap("c", pack.target_ids)
        misconceptions = AliasMap("m", [m.id for m in pack.misconceptions])
        earlier = view.turns[:-1][-self._transcript_turns :]
        try:
            out = await self._llm.generate(
                prompt_id=self.PROMPT_ID,
                variables={
                    "reference": fence_untrusted("reference", pack.reference_reasoning),
                    "concepts": concepts_block(pack.targets, concepts),
                    "misconceptions": misconceptions_block(
                        pack.misconceptions, concepts, misconceptions
                    ),
                    "transcript": transcript(earlier, max_answer_chars=self._max_answer_chars),
                    "question": fence_untrusted("question", latest.question_text),
                    "answer": fence_untrusted(
                        "student_answer", clip(latest.answer_text, self._max_answer_chars)
                    ),
                },
                output_model=ClassifyOut,
                ledger=ledger,
                policy=CallPolicy(timeout_s=timeout_s, max_attempts=1, max_repairs=0, lane="live"),
            )
        except LIVE_MODEL_FAILURES:
            return Classification.fallback()
        return normalize_labels(
            ModelLabels(
                answer_type=AnswerType(out.answer_type),
                confident=out.confidence == "high",
                misconception_ids=tuple(
                    misconceptions.resolve(alias)
                    for alias in out.misconceptions
                    if alias in misconceptions
                ),
                frustration=out.frustration,
                safety_concern=out.safety_concern,
                key_phrase=out.key_phrase,
            ),
            latest.answer_text,
        )
