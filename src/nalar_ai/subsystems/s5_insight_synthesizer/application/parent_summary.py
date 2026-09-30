"""Steps 3-4: the model writes a parent summary, code guards it (design doc §7).

The writer sees concept names, outcomes and the S4 summary only: no scores, flags, reference
reasoning or names (AI-8; parent content rules).
"""

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s5_insight_synthesizer.application.policy import SynthesizerPolicy
from nalar_ai.subsystems.s5_insight_synthesizer.application.rendering import concepts_block
from nalar_ai.subsystems.s5_insight_synthesizer.domain.config import SynthesizerConfig
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    ParentSummaryInput,
    check_summary,
)

TRIES = 2  # the first call and one fresh retry, then the template

SummarySource = Literal["model", "template"]

logger = logging.getLogger("nalar_ai.s5")


class SummaryDraftOut(BaseModel):
    summary: str


@dataclass(frozen=True, slots=True)
class ParentSummaryResult:
    content: str
    source: SummarySource
    retried: bool


class ParentSummaryUseCase:
    PROMPT_ID = "s5.parent_summary"

    def __init__(
        self, *, llm: LLMGateway, config: SynthesizerConfig, policy: SynthesizerPolicy
    ) -> None:
        self._llm = llm
        self._config = config
        self._policy = policy

    async def execute(self, data: ParentSummaryInput, ledger: UsageLedger) -> ParentSummaryResult:
        """Provider errors propagate (503): an outage must never freeze a template at release.
        Content problems end in the template."""
        variables = {
            "title": fence_untrusted("mission", data.mission_title),
            "concepts": concepts_block(data.concepts),
            "evaluation": fence_untrusted("teacher_summary", data.evaluation_summary)
            if data.evaluation_summary
            else "(none)",
        }
        names = data.names()
        feedback = "(none)"
        for attempt in range(TRIES):
            try:
                out = await self._llm.generate(
                    prompt_id=self.PROMPT_ID,
                    variables={**variables, "feedback": feedback},
                    output_model=SummaryDraftOut,
                    ledger=ledger,
                    policy=self._policy.summary_call(),
                )
            except OutputValidationError:
                problems = ["the answer was not valid JSON with a summary"]
            else:
                text = out.summary.strip()
                problems = check_summary(text, self._config.summary, names)
                if not problems:
                    return ParentSummaryResult(text, "model", retried=attempt > 0)
            # Problems name lexicon terms only, never the summary text.
            logger.warning(
                "s5 guard prompt=%s request_id=%s checks=%s",
                self.PROMPT_ID,
                ledger.request_id,
                "; ".join(problems),
            )
            feedback = "\n".join(f"- {problem}" for problem in problems)
        return ParentSummaryResult(
            self._config.template.render(data.concepts), "template", retried=True
        )
