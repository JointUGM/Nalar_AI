from typing import Annotated, Literal

from fastapi import APIRouter, Form, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.platform.http.uploads import read_pdf_upload
from nalar_ai.shared.enums import ChunkKind
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.subsystems.s1_knowledge_base.api.dependencies import EmbeddingsDep, S1Dep, TokensDep
from nalar_ai.subsystems.s1_knowledge_base.application.chunk_section import (
    ChunkSectionCommand,
    ChunkSectionUseCase,
)
from nalar_ai.subsystems.s1_knowledge_base.application.detect_sections import DetectSectionsUseCase

router = APIRouter()


class SectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ordinal: int
    level: int
    title: str
    page_start: int
    page_end: int
    parent_ordinal: int | None
    suggested: bool


class DetectSectionsOut(BaseModel):
    page_count: int
    has_toc: bool
    sections: list[SectionOut]
    pages_without_text: list[int]


@router.post("/sections/detect", response_model=Envelope[DetectSectionsOut])
async def detect_sections(
    file: UploadFile,
    s1: S1Dep,
    settings: SettingsDep,
    ledger: LedgerDep,
    fallback_title: Annotated[str, Form(min_length=1, max_length=300)] = "Materi",
) -> Envelope[DetectSectionsOut]:
    """Find the chapters of an uploaded PDF (TC-1). Makes no model calls.

    Build a level-1 section or its level-2 children, never both (they overlap).
    """
    pdf = await read_pdf_upload(file, settings.max_upload_bytes)
    result = await DetectSectionsUseCase(s1.reader, s1.lexicon).execute(
        pdf, fallback_title=fallback_title
    )
    return envelope(
        DetectSectionsOut(
            page_count=result.page_count,
            has_toc=result.has_toc,
            sections=[SectionOut.model_validate(section) for section in result.sections],
            pages_without_text=result.pages_without_text,
        ),
        ledger,
    )


class ChunkSpecIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=300)
    next_title: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _check_range(self) -> "ChunkSpecIn":
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self


class ChunkOut(BaseModel):
    local_index: int
    page_start: int
    page_end: int
    heading_path: str
    chunk_kind: ChunkKind
    content: str
    token_count: int
    content_sha256: str
    embedding: list[float]


class SkippedPageOut(BaseModel):
    page: int
    reason: Literal["scanned", "blank"]


class ChunkSectionOut(BaseModel):
    chunks: list[ChunkOut]
    skipped_pages: list[SkippedPageOut]
    embedding_model: str


@router.post("/sections/chunk", response_model=Envelope[ChunkSectionOut])
async def chunk_section(
    file: UploadFile,
    spec: Annotated[str, Form(description="JSON: {page_start, page_end, title, next_title}")],
    s1: S1Dep,
    settings: SettingsDep,
    ledger: LedgerDep,
    embeddings: EmbeddingsDep,
    tokens: TokensDep,
) -> Envelope[ChunkSectionOut]:
    """Parse, clean, structure, chunk, classify and embed one section (TC-1).

    Store each chunk with chunk_index = section.ordinal * 10_000 + local_index.
    """
    try:
        parsed = ChunkSpecIn.model_validate_json(spec)
    except ValidationError as exc:
        messages = [str(error["msg"]) for error in exc.errors()]
        raise InvalidInputError("invalid chunk spec", details={"errors": messages}) from exc
    pdf = await read_pdf_upload(file, settings.max_upload_bytes)
    use_case = ChunkSectionUseCase(
        reader=s1.reader,
        lexicon=s1.lexicon,
        embeddings=embeddings,
        count_tokens=tokens.count,
        policy=s1.chunking,
    )
    result = await use_case.execute(
        ChunkSectionCommand(
            pdf=pdf,
            page_start=parsed.page_start,
            page_end=parsed.page_end,
            title=parsed.title,
            next_title=parsed.next_title,
        ),
        ledger,
    )
    return envelope(
        ChunkSectionOut(
            chunks=[
                ChunkOut(
                    local_index=chunk.draft.local_index,
                    page_start=chunk.draft.page_start,
                    page_end=chunk.draft.page_end,
                    heading_path=chunk.draft.heading_path,
                    chunk_kind=chunk.draft.kind,
                    content=chunk.draft.content,
                    token_count=chunk.draft.token_count,
                    content_sha256=chunk.draft.content_sha256,
                    embedding=list(chunk.embedding),
                )
                for chunk in result.chunks
            ],
            skipped_pages=[
                SkippedPageOut(page=s.page, reason=s.reason) for s in result.skipped_pages
            ],
            embedding_model=result.embedding_model,
        ),
        ledger,
        result.warnings,
    )
