"""POST /v1/s5/parent-summaries/generate and its DTOs (design doc §4.3)."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.shared.enums import ConceptOutcome
from nalar_ai.subsystems.s5_insight_synthesizer.api.dependencies import GatewayDep, S5Dep
from nalar_ai.subsystems.s5_insight_synthesizer.application.parent_summary import (
    ParentSummaryUseCase,
    SummarySource,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    ParentConcept,
    ParentSummaryInput,
)

router = APIRouter()


class ParentConceptIn(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    outcome: ConceptOutcome = Field(description="session_concept_results.outcome")
    misconception_statement: str | None = Field(
        default=None,
        max_length=1000,
        description="The held idea, or the idea the child moved away from when resolved",
    )
    resolved_in_session: bool = False


class ParentSummaryIn(BaseModel):
    """One student's latest completed, evaluated attempt. No scores, flags, names or ids."""

    mission_title: str = Field(min_length=1, max_length=300)
    concepts: list[ParentConceptIn] = Field(min_length=1, max_length=10)
    evaluation_summary: str | None = Field(
        default=None, max_length=2000, description="session_evaluations.summary"
    )

    def to_input(self) -> ParentSummaryInput:
        return ParentSummaryInput(
            mission_title=self.mission_title,
            concepts=tuple(
                ParentConcept(c.name, c.outcome, c.misconception_statement, c.resolved_in_session)
                for c in self.concepts
            ),
            evaluation_summary=self.evaluation_summary,
        )


class ParentSummaryOut(BaseModel):
    content: str = Field(description="parent_summaries.content")
    source: SummarySource = Field(
        description="Not stored; 'template' after two blocked drafts (store ai_invocation_id null)"
    )


@router.post("/parent-summaries/generate", response_model=Envelope[ParentSummaryOut])
async def generate_parent_summary(
    body: ParentSummaryIn, ledger: LedgerDep, llm: GatewayDep, s5: S5Dep
) -> Envelope[ParentSummaryOut]:
    """Write one parent summary (TC-18, PA-1).

    Call once per eligible student; store the first success and never regenerate after release
    (AI-5). A provider outage is a 503, never a template.
    """
    use_case = ParentSummaryUseCase(llm=llm, config=s5.config, policy=s5.policy)
    result = await use_case.execute(body.to_input(), ledger)
    warnings = [
        *(["summary_retried"] if result.retried else []),
        *(["summary_template"] if result.source == "template" else []),
    ]
    return envelope(
        ParentSummaryOut(content=result.content, source=result.source), ledger, warnings
    )
