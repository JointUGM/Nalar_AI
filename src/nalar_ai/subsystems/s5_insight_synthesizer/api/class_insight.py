"""POST /v1/s5/class-insight and its DTOs (design doc §4.1, §4.2)."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.subsystems.s5_insight_synthesizer.api.dependencies import GatewayDep, S5Dep
from nalar_ai.subsystems.s5_insight_synthesizer.application.class_insight import (
    ClassInsightUseCase,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import (
    ClassCounts,
    ConceptCount,
    MisconceptionCount,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import Insight

router = APIRouter()

Count = Annotated[int, Field(ge=0, le=500)]


class MisconceptionCountIn(BaseModel):
    misconception_id: UUID
    statement: str = Field(min_length=1, max_length=1000)
    count: Count = Field(description="Students who hold it now")
    resolved_count: Count = Field(description="Students who changed their mind in the session")


class ConceptCountIn(BaseModel):
    concept_id: UUID
    name: str = Field(min_length=1, max_length=300)
    mastered_count: Count
    developing_count: Count
    not_observed_count: Count
    misconceptions: list[MisconceptionCountIn] = Field(default_factory=list, max_length=30)


class ClassInsightIn(BaseModel):
    """The backend's class-map counts (TC-9). Store exactly this as counts_snapshot."""

    mission_title: str = Field(min_length=1, max_length=300)
    denominator: int = Field(ge=1, le=500, description="Eligible latest attempts")
    incomplete_count: Count
    concepts: list[ConceptCountIn] = Field(min_length=1, max_length=10)

    def to_counts(self) -> ClassCounts:
        return ClassCounts(
            mission_title=self.mission_title,
            total=self.denominator,
            incomplete=self.incomplete_count,
            concepts=tuple(
                ConceptCount(
                    id=c.concept_id,
                    name=c.name,
                    mastered=c.mastered_count,
                    developing=c.developing_count,
                    not_observed=c.not_observed_count,
                    misconceptions=tuple(
                        MisconceptionCount(
                            m.misconception_id, m.statement, m.count, m.resolved_count
                        )
                        for m in c.misconceptions
                    ),
                )
                for c in self.concepts
            ),
        )


class ClusterOut(BaseModel):
    name: str
    explanation: str = Field(description="May contain placeholders, like the narrative")
    misconception_ids: list[UUID]


class ClassInsightOut(BaseModel):
    narrative: str = Field(
        description="class_map_insights.narrative. Placeholders: {{total}}, {{incomplete}}, "
        "{{count|resolved:<misconception_id>}}, "
        "{{mastered|developing|not_observed:<concept_id>}}"
    )
    clusters: list[ClusterOut] = Field(description="class_map_insights.clusters")

    @classmethod
    def from_insight(cls, insight: Insight) -> "ClassInsightOut":
        return cls(
            narrative=insight.narrative,
            clusters=[
                ClusterOut(
                    name=c.name,
                    explanation=c.explanation,
                    misconception_ids=list(c.misconception_ids),
                )
                for c in insight.clusters
            ],
        )


@router.post("/class-insight", response_model=Envelope[ClassInsightOut])
async def class_insight(
    body: ClassInsightIn, ledger: LedgerDep, llm: GatewayDep, s5: S5Dep
) -> Envelope[ClassInsightOut]:
    """Explain the class map for the teacher (TC-9), with placeholders instead of numbers (AI-4).

    Fill the placeholders from the counts you sent; never show the raw narrative. Fails with
    `ai_output_invalid` when two drafts fail the number guard: keep `insight` null and retry
    later.
    """
    use_case = ClassInsightUseCase(llm=llm, config=s5.config, policy=s5.policy)
    result = await use_case.execute(body.to_counts(), ledger)
    warnings = ["insight_retried"] if result.retried else []
    return envelope(ClassInsightOut.from_insight(result.insight), ledger, warnings)
