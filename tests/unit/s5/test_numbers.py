from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import (
    NumberLexicon,
    number_problems,
    without_names,
)

LEXICON = NumberLexicon.build(
    number_words=["satu", "dua", "setengah", "Kedua"],
    count_claims=["sebagian besar", "semua siswa"],
)


def test_clean_text_has_no_problems() -> None:
    assert number_problems("Siswa masih berpikir gaya bisa habis.", LEXICON) == []


def test_digits_are_rejected_including_non_ascii_digits() -> None:
    assert number_problems("Ada 12 siswa.", LEXICON) == ["write no digits"]
    assert number_problems("Ada ١٢ siswa.", LEXICON) == ["write no digits"]


def test_number_words_match_whole_words_only() -> None:
    assert number_problems("Dua siswa berubah pikiran.", LEXICON) == [
        "write no number words (found 'dua')"
    ]
    assert number_problems("Keduanya duar satuan", LEXICON) == []


def test_count_claims_are_rejected() -> None:
    assert number_problems("Sebagian besar siswa paham.", LEXICON) == [
        "do not say how many students with words (found 'sebagian besar'); use a placeholder"
    ]


def test_provided_names_are_exempt() -> None:
    names = ["Hukum Newton 1", "Satu Arah"]
    assert number_problems("Konsep hukum newton 1 dan Satu Arah sudah jelas.", LEXICON, names) == []
    assert without_names("Hukum Newton 1 x", names) == "  x"
