import pytest

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import (
    GENERATION_KINDS,
    KindLexicon,
    KindMarker,
    RegionScope,
)
from nalar_ai.subsystems.s1_knowledge_base.infrastructure.lexicon_loader import load_kind_lexicon

LEXICON = load_kind_lexicon()


@pytest.mark.parametrize(
    ("line", "bold", "kind"),
    [
        ("Ayo, Kita Lakukan", False, ChunkKind.ACTIVITY),
        ("Contoh Soal 1.2", False, ChunkKind.EXAMPLE),
        ("Uji Kompetensi Bab 1", False, ChunkKind.EXERCISE),
        ("Kunci Jawaban dan Pembahasan", False, ChunkKind.ANSWER_KEY),
        ("Tahukah Kamu?", True, ChunkKind.SIDEBAR),
        ("Rangkuman", True, ChunkKind.SUMMARY),
        ("Daftar Pustaka", False, ChunkKind.OTHER),
        ("Aktivitas 1.3 Mengukur Gaya Gesek", True, ChunkKind.ACTIVITY),
    ],
)
def test_markers_map_to_kinds(line: str, bold: bool, kind: ChunkKind) -> None:
    marker = LEXICON.match(line, bold=bold)
    assert marker is not None and marker.kind is kind


@pytest.mark.parametrize(
    "line",
    [
        "Contoh gaya gesek adalah gaya yang melawan gerak",
        "Informasi",
        "Aktivitas 1.3 Mengukur Gaya Gesek",
        "Latihan membuat otot kita menjadi lebih kuat dan sehat setiap hari",
    ],
)
def test_sentences_are_not_markers(line: str) -> None:
    assert LEXICON.match(line, bold=False) is None


def test_region_scopes() -> None:
    exercise = LEXICON.match("Uji Kompetensi", bold=True)
    activity = LEXICON.match("Ayo, Kita Lakukan", bold=True)
    assert exercise is not None and exercise.scope is RegionScope.SECTION
    assert activity is not None and activity.scope is RegionScope.BLOCK


def test_front_and_back_matter_titles() -> None:
    assert LEXICON.is_front_or_back_matter("Daftar Pustaka")
    assert LEXICON.is_front_or_back_matter("Kata Pengantar")
    assert not LEXICON.is_front_or_back_matter("Bab 1 Gaya dan Gerak")


def test_generation_kinds_never_include_restricted_kinds() -> None:
    assert set(GENERATION_KINDS) == {
        ChunkKind.EXPLANATION,
        ChunkKind.EXAMPLE,
        ChunkKind.ACTIVITY,
        ChunkKind.SUMMARY,
        ChunkKind.SIDEBAR,
    }


def test_explanation_cannot_have_markers() -> None:
    with pytest.raises(ValueError, match="default"):
        KindLexicon.build(1, [KindMarker(ChunkKind.EXPLANATION, RegionScope.BLOCK, "Materi")])


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("Uji Kompetensi Bab 1 Gaya dan Gerak", ChunkKind.EXERCISE),
        ("Kunci Jawaban Uji Kompetensi", ChunkKind.ANSWER_KEY),
        ("Latihan Soal Gaya", ChunkKind.EXERCISE),
    ],
)
def test_restrictive_markers_match_titled_lines_even_when_not_bold(
    line: str, kind: ChunkKind
) -> None:
    marker = LEXICON.match(line, bold=False)
    assert marker is not None and marker.kind is kind


@pytest.mark.parametrize("line", ["Indeks Bias", "Refleksi Cahaya", "Indeks Bias Cahaya pada Kaca"])
def test_back_matter_markers_do_not_swallow_physics_headings(line: str) -> None:
    assert LEXICON.match(line, bold=True) is None


@pytest.mark.parametrize(
    "title",
    [
        "Prakata",
        "Petunjuk Penggunaan Buku",
        "Daftar Gambar",
        "Daftar Tabel",
        "Lampiran",
        "Sampul",
        "Profil Pelaku Perbukuan",
    ],
)
def test_common_bse_front_and_back_matter_is_recognised(title: str) -> None:
    assert LEXICON.is_front_or_back_matter(title)
