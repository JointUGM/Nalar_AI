import pytest

from nalar_ai.shared.errors import DocumentUnreadableError, InvalidInputError
from nalar_ai.subsystems.s1_knowledge_base.domain.models import TocEntry
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.pymupdf_reader import PyMuPdfReader
from tests.support.pdf_factory import Line, ScannedPage, make_pdf

BODY = "Gaya adalah tarikan atau dorongan yang dapat mengubah gerak benda."


def _book() -> bytes:
    return make_pdf(
        [
            [Line("A. Gaya", size=14, bold=True), BODY],
            ScannedPage(),
            [],
            ["Bab 2 Tekanan", BODY],
        ],
        toc=[(1, "Bab 1 Gaya dan Gerak", 1), (2, "A. Gaya", 1), (1, "Bab 2 Tekanan", 4)],
    )


def test_inspect_reads_outline_and_pages_without_text() -> None:
    outline = PyMuPdfReader().inspect(_book())
    assert outline.page_count == 4
    assert outline.toc == (
        TocEntry(1, "Bab 1 Gaya dan Gerak", 1),
        TocEntry(2, "A. Gaya", 1),
        TocEntry(1, "Bab 2 Tekanan", 4),
    )
    assert outline.pages_without_text == (2, 3)


def test_read_pages_returns_lines_with_font_data() -> None:
    first, scanned = PyMuPdfReader().read_pages(_book(), 1, 2)
    heading, body = first.lines
    assert (heading.text, heading.font_size, heading.bold, heading.page) == (
        "A. Gaya",
        14.0,
        True,
        1,
    )
    assert (body.text, body.font_size, body.bold) == (BODY, 11.0, False)
    assert heading.y0 < body.y0
    assert first.image_coverage == 0.0
    assert scanned.lines == () and scanned.image_coverage == pytest.approx(1.0)


def test_page_range_is_validated() -> None:
    with pytest.raises(InvalidInputError):
        PyMuPdfReader().read_pages(_book(), 3, 9)


def test_garbage_is_unreadable() -> None:
    with pytest.raises(DocumentUnreadableError):
        PyMuPdfReader().inspect(b"%PDF-1.7 definitely not a pdf")
