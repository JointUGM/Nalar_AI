from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextLine:
    """One visual line of a PDF page, with geometry (points, origin top-left) and font data."""

    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float
    bold: bool


@dataclass(frozen=True, slots=True)
class PageContent:
    number: int  # 1-based
    width: float
    height: float
    lines: tuple[TextLine, ...]
    image_coverage: float = 0.0  # share of the page area covered by images, 0..1


@dataclass(frozen=True, slots=True)
class TocEntry:
    level: int
    title: str
    page: int  # 1-based


@dataclass(frozen=True, slots=True)
class PdfOutline:
    page_count: int
    toc: tuple[TocEntry, ...]
    pages_without_text: tuple[int, ...]
