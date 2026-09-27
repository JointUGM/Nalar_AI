import pytest

from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.shared.enums import ChunkKind
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.chunk_section import (
    ChunkSectionCommand,
    ChunkSectionUseCase,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.chunker import ChunkingPolicy
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.lexicon_loader import load_kind_lexicon
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.pymupdf_reader import PyMuPdfReader
from tests.support.pdf_factory import Line, ScannedPage, make_pdf


def _use_case() -> ChunkSectionUseCase:
    return ChunkSectionUseCase(
        reader=PyMuPdfReader(),
        lexicon=load_kind_lexicon(),
        embeddings=EmbeddingService(
            HashingEmbedder(), model="text-embedding-3-small", dimensions=1536
        ),
        count_tokens=lambda text: len(text.split()),
        policy=ChunkingPolicy(),
    )


async def test_chunks_are_embedded_with_their_heading_path() -> None:
    pdf = make_pdf(
        [
            [
                Line("Bab 1 Gaya dan Gerak", size=16, bold=True),
                Line("A. Gaya", size=14, bold=True, gap_before=6),
                "Gaya adalah tarikan atau dorongan pada benda.",
                Line("Rangkuman", bold=True, gap_before=12),
                "Gaya dapat mengubah gerak benda.",
            ],
            ScannedPage(),
        ]
    )
    ledger = UsageLedger("r", 1.0)
    result = await _use_case().execute(
        ChunkSectionCommand(pdf=pdf, page_start=1, page_end=2, title="Bab 1 Gaya dan Gerak"), ledger
    )
    assert [(c.draft.heading_path, c.draft.kind) for c in result.chunks] == [
        ("Bab 1 Gaya dan Gerak > A. Gaya", ChunkKind.EXPLANATION),
        ("Bab 1 Gaya dan Gerak > A. Gaya > Rangkuman", ChunkKind.SUMMARY),
    ]
    assert all(len(c.embedding) == 1536 for c in result.chunks)
    assert [(s.page, s.reason) for s in result.skipped_pages] == [(2, "scanned")]
    assert result.embedding_model == "text-embedding-3-small"
    assert result.warnings == ["1 page(s) without a text layer were skipped"]
    assert [r.prompt_version for r in ledger.records] == ["embed.chunk"]


async def test_a_section_without_text_returns_no_chunks_and_no_calls() -> None:
    ledger = UsageLedger("r", 1.0)
    result = await _use_case().execute(
        ChunkSectionCommand(pdf=make_pdf([[]]), page_start=1, page_end=1, title="Bab 1"), ledger
    )
    assert result.chunks == []
    assert result.warnings == ["the section produced no text chunks"]
    assert ledger.records == ()


async def test_page_range_outside_the_document_is_rejected() -> None:
    with pytest.raises(InvalidInputError):
        await _use_case().execute(
            ChunkSectionCommand(pdf=make_pdf([["teks"]]), page_start=1, page_end=5, title="Bab 1"),
            UsageLedger("r", 1.0),
        )
