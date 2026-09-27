from typing import Protocol

from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent, PdfOutline


class PdfReaderPort(Protocol):
    def inspect(self, data: bytes) -> PdfOutline: ...

    def read_pages(self, data: bytes, page_start: int, page_end: int) -> list[PageContent]: ...
