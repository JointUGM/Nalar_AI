"""Steps 1-2: the model explains the class map, code guards it (design doc §5, §6)."""

import logging
from dataclasses import dataclass

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.errors import InvalidInputError, OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s5_insight_synthesizer.application.policy import SynthesizerPolicy
from nalar_ai.subsystems.s5_insight_synthesizer.application.rendering import counts_block
from nalar_ai.subsystems.s5_insight_synthesizer.domain.config import SynthesizerConfig
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import ClassCounts, check_counts
from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import (
    DraftCluster,
    Insight,
    check_insight,
)

TRIES = 2  # the first call and one fresh retry, then 502
MAX_REPORTED = 10

logger = logging.getLogger("nalar_ai.s5")


class ClusterDraftOut(BaseModel):
    name: str
    explanation: str
    misconceptions: list[str]


class InsightDraftOut(BaseModel):
    narrative: str
    clusters: list[ClusterDraftOut]
    suggestions: list[str]


@dataclass(frozen=True, slots=True)
class InsightResult:
    insight: Insight
    retried: bool


class ClassInsightUseCase:
    PROMPT_ID = "s5.class_insight"

    def __init__(
        self, *, llm: LLMGateway, config: SynthesizerConfig, policy: SynthesizerPolicy
    ) -> None:
        self._llm = llm
        self._config = config
        self._policy = policy

    async def execute(self, counts: ClassCounts, ledger: UsageLedger) -> InsightResult:
        """Provider errors propagate (503); two guarded failures raise ai_output_invalid."""
        invalid = check_counts(counts)
        if invalid:
            raise InvalidInputError(
                "the class counts are inconsistent", details={"problems": invalid}
            )
        variables = {
            "title": fence_untrusted("mission", counts.mission_title),
            "total": str(counts.total),
            "incomplete": str(counts.incomplete),
            "counts": counts_block(counts),
        }
        feedback, problems = "(none)", ["no attempt"]
        for attempt in range(TRIES):
            try:
                out = await self._llm.generate(
                    prompt_id=self.PROMPT_ID,
                    variables={**variables, "feedback": feedback},
                    output_model=InsightDraftOut,
                    ledger=ledger,
                    policy=self._policy.insight_call(),
                )
            except OutputValidationError:
                problems = [
                    "the answer must be valid JSON with narrative, clusters and suggestions"
                ]
            else:
                drafts = [
                    DraftCluster(c.name, c.explanation, tuple(c.misconceptions))
                    for c in out.clusters
                ]
                insight, problems = check_insight(
                    out.narrative,
                    drafts,
                    counts,
                    self._config.numbers,
                    self._config.insight,
                    out.suggestions,
                )
                if insight is not None:
                    return InsightResult(insight, retried=attempt > 0)
            # Problems name aliases and lexicon terms only, never teacher or student text.
            logger.warning(
                "s5 guard prompt=%s request_id=%s checks=%s",
                self.PROMPT_ID,
                ledger.request_id,
                "; ".join(problems[:MAX_REPORTED]),
            )
            feedback = "\n".join(f"- {problem}" for problem in problems)
        raise OutputValidationError(
            f"{self.PROMPT_ID} failed the number guard {TRIES} times",
            details={"errors": problems[:MAX_REPORTED]},
        )
