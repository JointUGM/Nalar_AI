from dataclasses import dataclass

import anyio

from nalar_ai.subsystems.s1_knowledge_base.application.ports import PdfReaderPort
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import KindLexicon
from nalar_ai.subsystems.s1_knowledge_base.domain.sections import DetectedSection, detect_sections


@dataclass(frozen=True, slots=True)
class DetectSectionsResult:
    page_count: int
    has_toc: bool
    sections: list[DetectedSection]
    pages_without_text: list[int]


class DetectSectionsUseCase:
    def __init__(self, reader: PdfReaderPort, lexicon: KindLexicon) -> None:
        self._reader = reader
        self._lexicon = lexicon

    async def execute(self, pdf: bytes, *, fallback_title: str) -> DetectSectionsResult:
        outline = await anyio.to_thread.run_sync(self._reader.inspect, pdf)
        sections = detect_sections(
            outline,
            fallback_title=fallback_title,
            is_excluded_title=self._lexicon.is_front_or_back_matter,
        )
        return DetectSectionsResult(
            page_count=outline.page_count,
            has_toc=bool(outline.toc),
            sections=sections,
            pages_without_text=list(outline.pages_without_text),
        )
