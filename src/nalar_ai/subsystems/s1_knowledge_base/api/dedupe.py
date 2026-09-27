from typing import Literal
from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.subsystems.s1_knowledge_base.api.dependencies import GatewayDep
from nalar_ai.subsystems.s1_knowledge_base.application.dedupe_concepts import (
    DedupeConceptsUseCase,
    DedupItem,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.dedup_policy import (
    Candidate,
    DedupBasis,
    DedupThresholds,
)

router = APIRouter()


class CandidateIn(BaseModel):
    concept_id: UUID
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)
    similarity: float = Field(ge=-1.0, le=1.0)


class DedupItemIn(BaseModel):
    key: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)
    candidates: list[CandidateIn] = Field(default_factory=list, max_length=10)


class DedupeIn(BaseModel):
    items: list[DedupItemIn] = Field(min_length=1, max_length=100)


class DedupDecisionOut(BaseModel):
    key: str
    action: Literal["link", "create"]
    existing_concept_id: UUID | None
    basis: DedupBasis
    similarity: float | None
    reason: str | None


class DedupeOut(BaseModel):
    decisions: list[DedupDecisionOut]


@router.post("/concepts/dedupe", response_model=Envelope[DedupeOut])
async def dedupe_concepts(
    body: DedupeIn, settings: SettingsDep, ledger: LedgerDep, llm: GatewayDep
) -> Envelope[DedupeOut]:
    """Link or create each new concept, given match_kb_concepts candidates (kb_dedup)."""
    use_case = DedupeConceptsUseCase(
        llm=llm,
        thresholds=DedupThresholds(
            link_at=settings.s1_dedup_link_threshold,
            judge_from=settings.s1_dedup_judge_threshold,
        ),
    )
    decisions = await use_case.execute(
        [
            DedupItem(
                key=item.key,
                name=item.name,
                description=item.description,
                candidates=tuple(
                    Candidate(c.concept_id, c.name, c.description, c.similarity)
                    for c in item.candidates
                ),
            )
            for item in body.items
        ],
        ledger,
    )
    return envelope(
        DedupeOut(
            decisions=[
                DedupDecisionOut(
                    key=d.key,
                    action=d.action,
                    existing_concept_id=d.existing_concept_id,
                    basis=d.basis,
                    similarity=d.similarity,
                    reason=d.reason,
                )
                for d in decisions
            ]
        ),
        ledger,
    )
