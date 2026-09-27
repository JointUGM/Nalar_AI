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
from nalar_ai.subsystems.s1_knowledge_base.application.generate_misconceptions import (
    ConceptForMisconceptions,
    GenerateMisconceptionsCommand,
    GenerateMisconceptionsUseCase,
    LibraryCandidate,
)

router = APIRouter()


class LibraryCandidateIn(BaseModel):
    library_id: UUID
    statement: str = Field(min_length=1, max_length=2000)
    correct_understanding: str = Field(min_length=1, max_length=2000)
    student_phrasings: list[str] = Field(default_factory=list, max_length=20)
    counter_examples: list[str] = Field(default_factory=list, max_length=20)
    similarity: float = Field(ge=-1.0, le=1.0)


class ConceptForMisconceptionsIn(BaseModel):
    concept_ref: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)
    source_chunk_ids: list[UUID] = Field(min_length=1, max_length=50)
    library_candidates: list[LibraryCandidateIn] = Field(default_factory=list, max_length=10)
    existing_statements: list[str] = Field(default_factory=list, max_length=50)


class GenerateMisconceptionsIn(BaseModel):
    subject: str = Field(min_length=1, max_length=100)
    phase: str = Field(min_length=1, max_length=10)
    chunks: list[ChunkRefIn] = Field(min_length=1, max_length=2000)
    concepts: list[ConceptForMisconceptionsIn] = Field(min_length=1, max_length=30)


class MisconceptionOut(BaseModel):
    concept_ref: str
    statement: str
    correct_understanding: str
    detection_cues: list[str]
    counter_examples: list[str]
    source_chunk_ids: list[UUID]
    library_id: UUID | None
    embedding: list[float]


class FailedConceptOut(BaseModel):
    concept_ref: str
    error: str


class GenerateMisconceptionsOut(BaseModel):
    misconceptions: list[MisconceptionOut]
    dropped: list[DroppedOut]
    failed: list[FailedConceptOut]
    embedding_model: str


@router.post("/misconceptions/generate", response_model=Envelope[GenerateMisconceptionsOut])
async def generate_misconceptions(
    body: GenerateMisconceptionsIn,
    settings: SettingsDep,
    ledger: LedgerDep,
    llm: GatewayDep,
    embeddings: EmbeddingsDep,
    tokens: TokensDep,
) -> Envelope[GenerateMisconceptionsOut]:
    """Propose 1-4 misconceptions per new concept (kb_misconceptions).

    Send the chapter's chunks, and per concept its source chunks, match_misconception_library
    candidates and existing statements. Concepts in `failed` can be re-sent on their own.
    """
    use_case = GenerateMisconceptionsUseCase(
        llm=llm,
        embeddings=embeddings,
        estimate_tokens=tokens.claude_estimate,
        evidence_budget_tokens=settings.s1_evidence_pack_claude_tokens,
        duplicate_threshold=settings.s1_dedup_link_threshold,
    )
    result = await use_case.execute(
        GenerateMisconceptionsCommand(
            subject=body.subject,
            phase=body.phase,
            chunks=tuple(chunk.to_ref() for chunk in body.chunks),
            concepts=tuple(
                ConceptForMisconceptions(
                    concept_ref=c.concept_ref,
                    name=c.name,
                    description=c.description,
                    source_chunk_ids=tuple(c.source_chunk_ids),
                    library_candidates=tuple(
                        LibraryCandidate(
                            library_id=e.library_id,
                            statement=e.statement,
                            correct_understanding=e.correct_understanding,
                            student_phrasings=tuple(e.student_phrasings),
                            counter_examples=tuple(e.counter_examples),
                            similarity=e.similarity,
                        )
                        for e in c.library_candidates
                    ),
                    existing_statements=tuple(c.existing_statements),
                )
                for c in body.concepts
            ),
        ),
        ledger,
    )
    return envelope(
        GenerateMisconceptionsOut(
            misconceptions=[
                MisconceptionOut(
                    concept_ref=m.concept_ref,
                    statement=m.statement,
                    correct_understanding=m.correct_understanding,
                    detection_cues=list(m.detection_cues),
                    counter_examples=list(m.counter_examples),
                    source_chunk_ids=list(m.source_chunk_ids),
                    library_id=m.library_id,
                    embedding=list(m.embedding),
                )
                for m in result.misconceptions
            ],
            dropped=[DroppedOut(name=d.name, reason=d.reason) for d in result.dropped],
            failed=[
                FailedConceptOut(concept_ref=f.concept_ref, error=f.error) for f in result.failed
            ],
            embedding_model=result.embedding_model,
        ),
        ledger,
    )
