from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent
from nalar_ai.subsystems.s1_knowledge_base.domain.structure import (
    Paragraph,
    build_paragraphs,
    title_matches,
)
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.lexicon_loader import load_kind_lexicon
from tests.support.builders import L, make_page

LEXICON = load_kind_lexicon()
ROOT = "Bab 1 Gaya dan Gerak"
E, X, A, S, K = (
    ChunkKind.EXPLANATION,
    ChunkKind.EXERCISE,
    ChunkKind.ACTIVITY,
    ChunkKind.SIDEBAR,
    ChunkKind.ANSWER_KEY,
)


def _paragraphs(
    pages: list[PageContent], title: str = ROOT, next_title: str | None = None
) -> list[Paragraph]:
    return build_paragraphs(
        pages, section_title=title, next_section_title=next_title, lexicon=LEXICON
    )


def test_heading_path_follows_heading_levels() -> None:
    page = make_page(
        1,
        [
            L("A. Gaya", size=14, bold=True),
            "Gaya adalah tarikan atau dorongan.",
            L("1. Gaya Gesek", bold=True, gap=12),
            "Gaya gesek melawan gerak benda.",
            L("B. Gerak", size=14, bold=True, gap=12),
            "Gerak adalah perubahan posisi.",
        ],
    )
    assert [(p.heading_path, p.kind) for p in _paragraphs([page])] == [
        ((ROOT, "A. Gaya"), E),
        ((ROOT, "A. Gaya", "1. Gaya Gesek"), E),
        ((ROOT, "B. Gerak"), E),
    ]


def test_paragraphs_break_on_gaps_and_bullets() -> None:
    page = make_page(
        1,
        [
            "Baris satu paragraf pertama",
            "lanjutan paragraf pertama.",
            L("Paragraf kedua dimulai.", gap=12),
            "1) butir pertama",
            "2) butir kedua",
        ],
    )
    assert [p.text for p in _paragraphs([page])] == [
        "Baris satu paragraf pertama lanjutan paragraf pertama.",
        "Paragraf kedua dimulai.",
        "1) butir pertama",
        "2) butir kedua",
    ]


def test_paragraph_continues_across_pages() -> None:
    pages = [
        make_page(1, ["Gaya gesek terjadi ketika dua"]),
        make_page(2, ["permukaan bersentuhan."]),
    ]
    (paragraph,) = _paragraphs(pages)
    assert paragraph.text == "Gaya gesek terjadi ketika dua permukaan bersentuhan."
    assert (paragraph.page_start, paragraph.page_end) == (1, 2)


def test_exercise_region_and_answer_refinement() -> None:
    page = make_page(
        1,
        [
            L("A. Gaya", size=14, bold=True),
            "Penjelasan tentang gaya.",
            L("Uji Kompetensi", bold=True, gap=12),
            "1. Apa yang dimaksud dengan gaya?",
            L("Jawab: gaya adalah tarikan atau dorongan.", gap=12),
        ],
    )
    assert [(p.text, p.kind) for p in _paragraphs([page])] == [
        ("Penjelasan tentang gaya.", E),
        ("1. Apa yang dimaksud dengan gaya?", X),
        ("Jawab: gaya adalah tarikan atau dorongan.", K),
    ]


def test_exercise_region_survives_same_level_subheading() -> None:
    page = make_page(
        1,
        [
            L("Uji Kompetensi", bold=True),
            L("A. Pilihan Ganda", bold=True, gap=6),
            "1. Benda diam tidak memiliki gaya. Benar atau salah?",
            L("B. Uraian", bold=True, gap=6),
            "2. Jelaskan mengapa bola yang ditendang akhirnya berhenti.",
        ],
    )
    assert [p.kind for p in _paragraphs([page])] == [X, X]


def test_block_region_ends_at_next_heading() -> None:
    page = make_page(
        1,
        [
            L("A. Gaya", size=14, bold=True),
            L("Ayo, Kita Lakukan", bold=True, gap=6),
            "Dorong buku di atas meja lalu lepaskan.",
            L("B. Gerak", size=14, bold=True, gap=12),
            "Gerak adalah perubahan posisi.",
        ],
    )
    assert [(p.heading_path, p.kind) for p in _paragraphs([page])] == [
        ((ROOT, "A. Gaya", "Ayo, Kita Lakukan"), A),
        ((ROOT, "B. Gerak"), E),
    ]


def test_block_region_ends_after_a_large_gap() -> None:
    page = make_page(
        1,
        [
            L("A. Gaya", size=14, bold=True),
            L("Tahukah Kamu?", bold=True, gap=6),
            "Gesekan membuat kita bisa berjalan.",
            L("Gaya gesek juga menghasilkan panas.", gap=40),
        ],
    )
    assert [(p.heading_path, p.kind) for p in _paragraphs([page])] == [
        ((ROOT, "A. Gaya", "Tahukah Kamu?"), S),
        ((ROOT, "A. Gaya"), E),
    ]


def test_marker_word_inside_a_sentence_is_explanation() -> None:
    page = make_page(1, ["Contoh gaya gesek adalah gaya yang melawan gerak."])
    assert [p.kind for p in _paragraphs([page])] == [E]


def test_section_is_trimmed_to_its_own_title_and_the_next_title() -> None:
    page1 = make_page(
        1,
        [
            "Akhir bab sebelumnya yang tidak ikut.",
            L("Bab 2", size=16, bold=True, gap=12),
            L("Tekanan Zat", size=16, bold=True),
            "Tekanan adalah gaya per satuan luas.",
        ],
    )
    page2 = make_page(
        2,
        [
            "Tekanan zat cair bergantung pada kedalaman.",
            L("Bab 3", size=16, bold=True, gap=12),
            L("Getaran dan Gelombang", size=16, bold=True),
            "Getaran adalah gerak bolak-balik.",
        ],
    )
    (paragraph,) = _paragraphs(
        [page1, page2], title="Bab 2 Tekanan Zat", next_title="Bab 3 Getaran dan Gelombang"
    )
    assert paragraph.text == (
        "Tekanan adalah gaya per satuan luas. Tekanan zat cair bergantung pada kedalaman."
    )
    assert paragraph.heading_path == ("Bab 2 Tekanan Zat",)


def test_title_not_found_keeps_everything() -> None:
    page = make_page(1, ["Teks pembuka materi ini.", "Kalimat kedua."])
    (paragraph,) = _paragraphs([page], title="Judul Lain Sekali")
    assert paragraph.text == "Teks pembuka materi ini. Kalimat kedua."


def test_title_matching() -> None:
    assert title_matches("Bab 2", "Bab 2 Tekanan Zat")
    assert title_matches("Tekanan Zat", "Bab 2 Tekanan Zat")
    assert not title_matches("Bab 1", "Bab 10 Listrik")
    assert not title_matches("Zat", "Bab 2 Tekanan Zat")
