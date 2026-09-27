from nalar_ai.subsystems.s1_knowledge_base.domain.models import PdfOutline, TocEntry
from nalar_ai.subsystems.s1_knowledge_base.domain.sections import detect_sections
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.lexicon_loader import load_kind_lexicon

LEXICON = load_kind_lexicon()


def _outline(toc: list[tuple[int, str, int]], pages: int = 40) -> PdfOutline:
    return PdfOutline(pages, tuple(TocEntry(*entry) for entry in toc), ())


def _rows(outline: PdfOutline) -> list[tuple[object, ...]]:
    return [
        (s.ordinal, s.level, s.title, s.page_start, s.page_end, s.parent_ordinal, s.suggested)
        for s in detect_sections(
            outline, fallback_title="Materi", is_excluded_title=LEXICON.is_front_or_back_matter
        )
    ]


def test_sections_come_from_the_outline_with_inclusive_page_ends() -> None:
    toc = [
        (1, "Kata Pengantar", 1),
        (1, "Bab 1 Gaya dan Gerak", 3),
        (2, "A. Gaya", 3),
        (2, "B. Gerak", 10),
        (3, "1. Kecepatan", 11),
        (1, "Bab 2 Tekanan Zat", 20),
        (1, "Daftar Pustaka", 39),
    ]
    assert _rows(_outline(toc)) == [
        (1, 1, "Kata Pengantar", 1, 3, None, False),
        (2, 1, "Bab 1 Gaya dan Gerak", 3, 20, None, True),
        (3, 2, "A. Gaya", 3, 10, 2, False),
        (4, 2, "B. Gerak", 10, 20, 2, False),
        (5, 1, "Bab 2 Tekanan Zat", 20, 39, None, True),
        (6, 1, "Daftar Pustaka", 39, 40, None, False),
    ]


def test_no_outline_means_one_section() -> None:
    assert _rows(_outline([], pages=12)) == [(1, 1, "Materi", 1, 12, None, True)]


def test_outlines_starting_at_level_two_are_normalised() -> None:
    rows = _rows(_outline([(2, "A. Gaya", 1), (2, "B. Gerak", 5)], pages=9))
    assert rows == [(1, 1, "A. Gaya", 1, 5, None, True), (2, 1, "B. Gerak", 5, 9, None, True)]
