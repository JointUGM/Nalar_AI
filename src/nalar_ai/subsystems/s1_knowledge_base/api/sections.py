from typing import Annotated

from fastapi import APIRouter, Form, UploadFile
from pydantic import BaseModel, ConfigDict

from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.platform.http.uploads import read_pdf_upload
from nalar_ai.subsystems.s1_knowledge_base.api.dependencies import S1Dep
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
