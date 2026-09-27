from nalar_ai.subsystems.s1_knowledge_base.domain.layout import (
    empty_page_reason,
    join_lines,
    reading_order,
    strip_running_furniture,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent
from tests.support.builders import line, make_page

BODY = "Gaya gesek selalu berlawanan arah dengan gerak benda."


def _page_with_furniture(number: int) -> PageContent:
    return make_page(
        number,
        [BODY, "Kalimat unik halaman ini."],
        extra=(
            line("Bab 1 | Gaya dan Gerak", y=30, page=number),
            line(str(10 + number), y=800, page=number),
        ),
    )


def test_running_headers_and_page_numbers_are_removed() -> None:
    pages = [_page_with_furniture(n) for n in (1, 2, 3, 4)]
    cleaned = strip_running_furniture(pages)
    for page in cleaned:
        assert [ln.text for ln in page.lines] == [BODY, "Kalimat unik halaman ini."]


def test_repeated_body_text_is_kept() -> None:
    pages = [make_page(n, [BODY]) for n in (1, 2, 3, 4)]
    assert all(page.lines for page in strip_running_furniture(pages))


def test_page_numbers_are_removed_even_on_a_single_page() -> None:
    page = make_page(1, [BODY], extra=(line("hlm. 12", y=805, page=1),))
    assert [ln.text for ln in strip_running_furniture([page])[0].lines] == [BODY]


def test_two_column_pages_read_left_column_first() -> None:
    lines = (
        line("kanan satu", y=100, x=320, width=200),
        line("kiri satu", y=100, x=72, width=200),
        line("kanan dua", y=120, x=320, width=200),
        line("kiri dua", y=120, x=72, width=200),
    )
    page = make_page(1, [], extra=lines)
    assert [ln.text for ln in reading_order(page)] == [
        "kiri satu",
        "kiri dua",
        "kanan satu",
        "kanan dua",
    ]


def test_single_column_pages_read_top_to_bottom() -> None:
    page = make_page(1, ["satu", "dua", "tiga"])
    assert [ln.text for ln in reading_order(page)] == ["satu", "dua", "tiga"]


def test_join_lines_dehyphenates_only_mid_word_breaks() -> None:
    assert (
        join_lines(["gaya gesek-", "an terjadi", "pada benda"]) == "gaya gesekan terjadi pada benda"
    )
    assert join_lines(["Newton -", "Hukum I"]) == "Newton - Hukum I"


def test_empty_page_reasons() -> None:
    assert empty_page_reason(make_page(1, [BODY])) is None
    assert empty_page_reason(make_page(2, [], image_coverage=0.9)) == "scanned"
    assert empty_page_reason(make_page(3, [])) == "blank"
