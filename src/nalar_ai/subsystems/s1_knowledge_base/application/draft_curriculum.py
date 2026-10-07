"""CP draft (cp_extract): copy Capaian Pembelajaran verbatim out of one excerpt of an official document.

Verification against the full document and merging across excerpts belong to the backend, which holds
every page (D-CPD-2). The model copies each sentence once, as pieces; the element text and statements
are rebuilt here so the response keeps its shape and the output is half as long (D-CPD-16).
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from nalar_ai.platform.llm.gateway import CallPolicy, LLMGateway
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted


class PrintedQuote(BaseModel):
    text: str = Field(max_length=300)
    page: int


class CpStatement(BaseModel):
    text: str = Field(max_length=4000)
    page_start: int
    page_end: int


class CpElement(BaseModel):
    element: str = Field(max_length=200)
    text: str = Field(max_length=30_000)
    page_start: int
    page_end: int
    statements: list[CpStatement]


class CpSubject(BaseModel):
    name: str = Field(max_length=200)
    phase: Literal["A", "B", "C", "D", "E", "F"]
    elements: list[CpElement]


class CpExcerptDraft(BaseModel):
    decree_code: PrintedQuote | None
    effective_on: PrintedQuote | None
    subjects: list[CpSubject]


class _Piece(BaseModel):
    text: str = Field(max_length=4000)
    page_start: int
    page_end: int
    statement: bool


class _PiecedElement(BaseModel):
    element: str = Field(max_length=200)
    pieces: list[_Piece]


class _PiecedSubject(BaseModel):
    name: str = Field(max_length=200)
    phase: Literal["A", "B", "C", "D", "E", "F"]
    elements: list[_PiecedElement]


class _PiecedDraft(BaseModel):
    """What the model writes: each element's outcome text once, split into pieces."""

    decree_code: PrintedQuote | None
    effective_on: PrintedQuote | None
    subjects: list[_PiecedSubject]


@dataclass(frozen=True, slots=True)
class DraftCurriculumCommand:
    title: str
    pages: Mapping[int, str]


class DraftCurriculumUseCase:
    PROMPT_ID = "s1.cp_draft"

    def __init__(self, *, llm: LLMGateway, timeout_s: float) -> None:
        self._llm = llm
        # One attempt, so the whole call stays inside the backend's S1 step timeout; the backend
        # requeues, and an abandoned retry here would spend money its ledger never sees (AI-6).
        self._policy = CallPolicy(timeout_s=timeout_s, max_attempts=1)

    async def execute(self, command: DraftCurriculumCommand, ledger: UsageLedger) -> CpExcerptDraft:
        excerpt = "\n\n".join(f"[page {n}]\n{text}" for n, text in sorted(command.pages.items()))
        allowed = set(command.pages)
        draft = await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "title": fence_untrusted("title", command.title),
                "excerpt": fence_untrusted("document", excerpt),
            },
            output_model=_PiecedDraft,
            ledger=ledger,
            semantic_check=lambda out: _page_errors(out, allowed),
            policy=self._policy,
        )
        return _rebuilt(draft)


def _rebuilt(draft: _PiecedDraft) -> CpExcerptDraft:
    return CpExcerptDraft(
        decree_code=draft.decree_code,
        effective_on=draft.effective_on,
        subjects=[
            CpSubject(
                name=subject.name,
                phase=subject.phase,
                elements=[
                    CpElement(
                        element=element.element,
                        # Pieces are consecutive, so a space join is the paragraph after the
                        # whitespace normalisation the backend's verbatim check applies.
                        text=" ".join(piece.text for piece in element.pieces),
                        page_start=min(piece.page_start for piece in element.pieces),
                        page_end=max(piece.page_end for piece in element.pieces),
                        statements=[
                            CpStatement(
                                text=piece.text,
                                page_start=piece.page_start,
                                page_end=piece.page_end,
                            )
                            for piece in element.pieces
                            if piece.statement
                        ],
                    )
                    for element in subject.elements
                    if element.pieces
                ],
            )
            for subject in draft.subjects
        ],
    )


def _cited(draft: _PiecedDraft) -> Iterable[tuple[int, int]]:
    for quote in (draft.decree_code, draft.effective_on):
        if quote is not None:
            yield quote.page, quote.page
    for subject in draft.subjects:
        for element in subject.elements:
            for piece in element.pieces:
                yield piece.page_start, piece.page_end


def _page_errors(draft: _PiecedDraft, allowed: set[int]) -> list[str]:
    return [
        f"pages {start}-{end} are not in the excerpt; cite only [page N] markers you were given"
        for start, end in _cited(draft)
        if start not in allowed or end not in allowed or end < start
    ][:5]
