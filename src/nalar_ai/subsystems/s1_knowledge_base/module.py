"""S1 composition: the subsystem's long-lived collaborators, built once from settings."""

from dataclasses import dataclass
from pathlib import Path

from nalar_ai.settings import Settings
from nalar_ai.subsystems.s1_knowledge_base.application.ports import PdfReaderPort
from nalar_ai.subsystems.s1_knowledge_base.domain.chunker import ChunkingPolicy
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import KindLexicon
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.lexicon_loader import load_kind_lexicon
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.pymupdf_reader import PyMuPdfReader

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True, slots=True)
class S1Module:
    reader: PdfReaderPort
    lexicon: KindLexicon
    chunking: ChunkingPolicy


def build_s1_module(settings: Settings) -> S1Module:
    return S1Module(
        reader=PyMuPdfReader(),
        lexicon=load_kind_lexicon(),
        chunking=ChunkingPolicy(
            max_tokens=settings.s1_chunk_max_tokens,
            min_tail_tokens=settings.s1_chunk_min_tail_tokens,
            hard_max_tokens=settings.s1_chunk_hard_max_tokens,
        ),
    )
