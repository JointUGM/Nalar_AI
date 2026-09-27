from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pymupdf

from nalar_ai.shared.errors import DocumentUnreadableError, InvalidInputError
from nalar_ai.shared.text import normalize_text
from nalar_ai.subsystems.s1_knowledge_base.domain.models import (
    PageContent,
    PdfOutline,
    TextLine,
    TocEntry,
)

MIN_TEXT_CHARS = 20
_BOLD_FLAG = 16  # PyMuPDF span flag bit 4


class PyMuPdfReader:
    """PdfReaderPort over PyMuPDF. CPU-bound: call it through anyio.to_thread."""

    def inspect(self, data: bytes) -> PdfOutline:
        with _open(data) as doc:
            toc = tuple(
                TocEntry(level=int(level), title=normalize_text(str(title)), page=int(page))
                for level, title, page, *_ in doc.get_toc(simple=True)
                if int(page) >= 1
            )
            without_text = tuple(
                number
                for number, page in enumerate(doc, start=1)
                if len(page.get_text("text").strip()) < MIN_TEXT_CHARS
            )
            return PdfOutline(
                page_count=int(doc.page_count), toc=toc, pages_without_text=without_text
            )

    def read_pages(self, data: bytes, page_start: int, page_end: int) -> list[PageContent]:
        with _open(data) as doc:
            if not 1 <= page_start <= page_end <= doc.page_count:
                raise InvalidInputError(
                    f"page range {page_start}-{page_end} is outside 1-{doc.page_count}"
                )
            return [
                _read_page(doc[number - 1], number) for number in range(page_start, page_end + 1)
            ]


@contextmanager
def _open(data: bytes) -> Iterator[Any]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentUnreadableError("the PDF could not be opened") from exc
    try:
        if doc.needs_pass:
            raise DocumentUnreadableError("the PDF is password protected")
        if doc.page_count == 0:
            raise DocumentUnreadableError("the PDF has no pages")
        yield doc
    finally:
        doc.close()


def _read_page(page: Any, number: int) -> PageContent:
    lines: list[TextLine] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for raw_line in block["lines"]:
            spans = [span for span in raw_line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = normalize_text("".join(span["text"] for span in raw_line["spans"]))
            if not text:
                continue
            x0, y0, x1, y1 = (float(value) for value in raw_line["bbox"])
            lines.append(
                TextLine(
                    text=text,
                    page=number,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    font_size=round(max(float(span["size"]) for span in spans), 1),
                    bold=all(
                        bool(int(span["flags"]) & _BOLD_FLAG) or "bold" in str(span["font"]).lower()
                        for span in spans
                    ),
                )
            )
    width, height = float(page.rect.width), float(page.rect.height)
    return PageContent(
        number=number,
        width=width,
        height=height,
        lines=tuple(lines),
        image_coverage=_image_coverage(page, width * height),
    )


def _image_coverage(page: Any, area: float) -> float:
    if area <= 0:
        return 0.0
    covered = 0.0
    for image in page.get_images(full=True):
        for rect in page.get_image_rects(image[0]):
            covered += float(rect.width) * float(rect.height)
    return min(1.0, covered / area)
