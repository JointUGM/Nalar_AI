from fastapi import APIRouter
from pydantic import BaseModel, Field

from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.shared.errors import InvalidInputError, PayloadTooLargeError
from nalar_ai.subsystems.s1_knowledge_base.api.dependencies import GatewayDep
from nalar_ai.subsystems.s1_knowledge_base.application.draft_curriculum import (
    CpExcerptDraft,
    DraftCurriculumCommand,
    DraftCurriculumUseCase,
)

router = APIRouter()


class ExcerptPageIn(BaseModel):
    page_number: int = Field(ge=1, le=500)
    text: str = Field(max_length=40_000)


class ExtractCurriculumIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    pages: list[ExcerptPageIn] = Field(min_length=1, max_length=40)


@router.post("/curriculum/extract", response_model=Envelope[CpExcerptDraft])
async def extract_curriculum(
    body: ExtractCurriculumIn, settings: SettingsDep, ledger: LedgerDep, llm: GatewayDep
) -> Envelope[CpExcerptDraft]:
    """Copy CP subjects, elements and statements verbatim from one excerpt (cp_extract)."""
    pages = {page.page_number: page.text for page in body.pages}
    if len(pages) != len(body.pages):
        raise InvalidInputError("page numbers must be distinct")
    size = sum(len(text) for text in pages.values())
    if size > settings.s1_cp_excerpt_max_chars:
        raise PayloadTooLargeError(
            "excerpt is too long",
            details={"characters": size, "limit": settings.s1_cp_excerpt_max_chars},
        )
    draft = await DraftCurriculumUseCase(
        llm=llm, timeout_s=settings.s1_cp_draft_timeout_seconds
    ).execute(DraftCurriculumCommand(body.title, pages), ledger)
    return envelope(draft, ledger)
