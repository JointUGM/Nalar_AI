from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.subsystems.s1_knowledge_base.api.dependencies import GatewayDep
from nalar_ai.subsystems.s1_knowledge_base.application.align_cp import (
    AlignCpUseCase,
    AlignItem,
    CpBasis,
    CpCandidate,
)

router = APIRouter()


class CpCandidateIn(BaseModel):
    outcome_id: UUID
    element: str | None = Field(default=None, max_length=300)
    description: str = Field(min_length=1, max_length=4000)
    similarity: float = Field(ge=-1.0, le=1.0)


class AlignItemIn(BaseModel):
    concept_ref: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)
    candidates: list[CpCandidateIn] = Field(default_factory=list, max_length=10)


class AlignCpIn(BaseModel):
    items: list[AlignItemIn] = Field(min_length=1, max_length=100)


class CpAlignmentOut(BaseModel):
    concept_ref: str
    outcome_id: UUID | None
    basis: CpBasis
    reason: str


class AlignCpOut(BaseModel):
    alignments: list[CpAlignmentOut]


@router.post("/concepts/align-cp", response_model=Envelope[AlignCpOut])
async def align_cp(
    body: AlignCpIn, settings: SettingsDep, ledger: LedgerDep, llm: GatewayDep
) -> Envelope[AlignCpOut]:
    """Label each concept with one CP statement from match_cp_outcomes candidates (cp_align).

    Also used to re-align concepts listed in v_concepts_needing_cp_realign after an SA-7 remap.
    """
    use_case = AlignCpUseCase(llm=llm, min_similarity=settings.s1_cp_min_similarity)
    alignments = await use_case.execute(
        [
            AlignItem(
                concept_ref=item.concept_ref,
                name=item.name,
                description=item.description,
                candidates=tuple(
                    CpCandidate(c.outcome_id, c.element, c.description, c.similarity)
                    for c in item.candidates
                ),
            )
            for item in body.items
        ],
        ledger,
    )
    return envelope(
        AlignCpOut(
            alignments=[
                CpAlignmentOut(
                    concept_ref=a.concept_ref,
                    outcome_id=a.outcome_id,
                    basis=a.basis,
                    reason=a.reason,
                )
                for a in alignments
            ]
        ),
        ledger,
    )
