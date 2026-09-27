from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import anyio

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.ports import PdfReaderPort
from nalar_ai.subsystems.s1_knowledge_base.domain.chunker import (
    ChunkDraft,
    ChunkingPolicy,
    chunk_paragraphs,
    embedding_text,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import KindLexicon
from nalar_ai.subsystems.s1_knowledge_base.domain.layout import (
    empty_page_reason,
    strip_running_furniture,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent
from nalar_ai.subsystems.s1_knowledge_base.domain.structure import build_paragraphs


@dataclass(frozen=True, slots=True)
class ChunkSectionCommand:
    pdf: bytes
    page_start: int
    page_end: int
    title: str
    next_title: str | None = None  # next section at the same or a higher level, for trimming


@dataclass(frozen=True, slots=True)
class SkippedPage:
    page: int
    reason: Literal["scanned", "blank"]


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    draft: ChunkDraft
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ChunkSectionResult:
    chunks: list[EmbeddedChunk]
    skipped_pages: list[SkippedPage]
    embedding_model: str
    warnings: list[str]


class ChunkSectionUseCase:
    def __init__(
        self,
        *,
        reader: PdfReaderPort,
        lexicon: KindLexicon,
        embeddings: EmbeddingService,
        count_tokens: Callable[[str], int],
        policy: ChunkingPolicy,
    ) -> None:
        self._reader = reader
        self._lexicon = lexicon
        self._embeddings = embeddings
        self._count_tokens = count_tokens
        self._policy = policy

    async def execute(
        self, command: ChunkSectionCommand, ledger: UsageLedger
    ) -> ChunkSectionResult:
        if command.page_end < command.page_start:
            raise InvalidInputError("page_end must be >= page_start")
        pages = await anyio.to_thread.run_sync(
            self._reader.read_pages, command.pdf, command.page_start, command.page_end
        )
        skipped: list[SkippedPage] = []
        readable: list[PageContent] = []
        for page in pages:
            reason = empty_page_reason(page)
            if reason is None:
                readable.append(page)
            else:
                skipped.append(SkippedPage(page.number, reason))
        drafts = await anyio.to_thread.run_sync(self._chunk, readable, command)
        scanned = sum(1 for page in skipped if page.reason == "scanned")
        warnings = [f"{scanned} page(s) without a text layer were skipped"] if scanned else []
        if not drafts:
            warnings.append("the section produced no text chunks")
            return ChunkSectionResult([], skipped, self._embeddings.model, warnings)
        vectors = await self._embeddings.embed(
            [embedding_text(draft.heading_path, draft.content) for draft in drafts],
            tag="chunk",
            ledger=ledger,
        )
        return ChunkSectionResult(
            chunks=[EmbeddedChunk(d, v) for d, v in zip(drafts, vectors, strict=True)],
            skipped_pages=skipped,
            embedding_model=self._embeddings.model,
            warnings=warnings,
        )

    def _chunk(self, pages: list[PageContent], command: ChunkSectionCommand) -> list[ChunkDraft]:
        paragraphs = build_paragraphs(
            strip_running_furniture(pages),
            section_title=command.title,
            next_section_title=command.next_title,
            lexicon=self._lexicon,
        )
        return chunk_paragraphs(paragraphs, self._count_tokens, self._policy)
