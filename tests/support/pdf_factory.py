"""Build small PDFs in memory, so tests never depend on real textbooks (copyright, size)."""

from collections.abc import Sequence
from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class Line:
    text: str
    size: float = 11.0
    bold: bool = False
    x: float = 72.0
    gap_before: float = 0.0
    y: float | None = None  # absolute baseline; None flows below the previous line


@dataclass(frozen=True)
class ScannedPage:
    """A page that is one image with no text layer."""


PageSpec = Sequence[Line | str] | ScannedPage


def make_pdf(pages: Sequence[PageSpec], *, toc: Sequence[tuple[int, str, int]] = ()) -> bytes:
    doc = pymupdf.open()
    try:
        for spec in pages:
            page = doc.new_page(width=595, height=842)
            if isinstance(spec, ScannedPage):
                pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
                pixmap.clear_with(180)
                page.insert_image(page.rect, pixmap=pixmap)
                continue
            y = 100.0
            for item in spec:
                line = Line(item) if isinstance(item, str) else item
                y += line.gap_before
                baseline = line.y if line.y is not None else y
                page.insert_text(
                    (line.x, baseline),
                    line.text,
                    fontsize=line.size,
                    fontname="hebo" if line.bold else "helv",
                )
                y = baseline + line.size * 1.5
        if toc:
            doc.set_toc([list(entry) for entry in toc])
        return bytes(doc.tobytes())
    finally:
        doc.close()
