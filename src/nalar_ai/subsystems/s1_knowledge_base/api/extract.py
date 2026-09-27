from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.subsystems.s1_knowledge_base.api.common import ChunkRefIn, DroppedOut
from nalar_ai.subsystems.s1_knowledge_base.api.dependencies import (
    EmbeddingsDep,
    GatewayDep,
    TokensDep,
)
from nalar_ai.subsystems.s1_knowledge_base.application.extract_concepts import (
    ExistingConcept,
    ExtractConceptsCommand,
    ExtractConceptsUseCase,
    PrerequisiteEdge,
    RejectedConcept,
)

router = APIRouter()


class ExistingConceptIn(BaseModel):
    id: UUID
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)


class PrerequisiteEdgeIn(BaseModel):
    concept_id: UUID
    prerequisite_concept_id: UUID


class RejectedConceptIn(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)


class ExtractConceptsIn(BaseModel):
    subject: str = Field(min_length=1, max_length=100)
    phase: str = Field(min_length=1, max_length=10)
    section_title: str = Field(min_length=1, max_length=300)
    chunks: list[ChunkRefIn] = Field(min_length=1, max_length=2000)
    existing_concepts: list[ExistingConceptIn] = Field(default_factory=list, max_length=1000)
    existing_prerequisites: list[PrerequisiteEdgeIn] = Field(default_factory=list, max_length=5000)
    rejected_concepts: list[RejectedConceptIn] = Field(default_factory=list, max_length=1000)


class ConceptDraftOut(BaseModel):
    key: str
    name: str
    description: str
    source_chunk_ids: list[UUID]
    prerequisite_keys: list[str]
    prerequisite_existing_ids: list[UUID]
    embedding: list[float]


class ExistingLinkOut(BaseModel):
    concept_id: UUID
    source_chunk_ids: list[UUID]


class ExtractConceptsOut(BaseModel):
    concepts: list[ConceptDraftOut]
    existing_links: list[ExistingLinkOut]
    dropped: list[DroppedOut]
    embedding_model: str


@router.post("/concepts/extract", response_model=Envelope[ExtractConceptsOut])
async def extract_concepts(
    body: ExtractConceptsIn,
    settings: SettingsDep,
    ledger: LedgerDep,
    llm: GatewayDep,
    embeddings: EmbeddingsDep,
    tokens: TokensDep,
) -> Envelope[ExtractConceptsOut]:
    """List a chapter's key concepts and prerequisites from its chunks (TC-1, kb_extract).

    Send the chapter's stored chunks, the knowledge base's existing and rejected concepts and
    existing prerequisite edges. New concepts come back with embeddings for match_kb_concepts.
    """
    use_case = ExtractConceptsUseCase(
        llm=llm,
        embeddings=embeddings,
        estimate_tokens=tokens.claude_estimate,
        max_section_tokens=settings.s1_max_section_claude_tokens,
        duplicate_threshold=settings.s1_dedup_link_threshold,
    )
    result = await use_case.execute(
        ExtractConceptsCommand(
            subject=body.subject,
            phase=body.phase,
            section_title=body.section_title,
            chunks=tuple(chunk.to_ref() for chunk in body.chunks),
            existing_concepts=tuple(
                ExistingConcept(c.id, c.name, c.description) for c in body.existing_concepts
            ),
            existing_prerequisites=tuple(
                PrerequisiteEdge(e.concept_id, e.prerequisite_concept_id)
                for e in body.existing_prerequisites
            ),
            rejected_concepts=tuple(
                RejectedConcept(r.name, r.description) for r in body.rejected_concepts
            ),
        ),
        ledger,
    )
    return envelope(
        ExtractConceptsOut(
            concepts=[
                ConceptDraftOut(
                    key=d.key,
                    name=d.name,
                    description=d.description,
                    source_chunk_ids=list(d.source_chunk_ids),
                    prerequisite_keys=list(d.prerequisite_keys),
                    prerequisite_existing_ids=list(d.prerequisite_existing_ids),
                    embedding=list(d.embedding),
                )
                for d in result.concepts
            ],
            existing_links=[
                ExistingLinkOut(
                    concept_id=link.concept_id, source_chunk_ids=list(link.source_chunk_ids)
                )
                for link in result.existing_links
            ],
            dropped=[DroppedOut(name=d.name, reason=d.reason) for d in result.dropped],
            embedding_model=result.embedding_model,
        ),
        ledger,
        result.warnings,
    )
