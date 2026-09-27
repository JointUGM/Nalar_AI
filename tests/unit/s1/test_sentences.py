from nalar_ai.subsystems.s1_knowledge_base.domain.sentences import split_sentences


def test_splits_on_sentence_ends() -> None:
    assert split_sentences("Gaya adalah dorongan. Gaya dapat mengubah gerak! Mengapa begitu?") == [
        "Gaya adalah dorongan.",
        "Gaya dapat mengubah gerak!",
        "Mengapa begitu?",
    ]


def test_keeps_abbreviations_decimals_and_quotes_together() -> None:
    text = 'Lihat Gb. 1.2 dan dll. Percepatan 9.8 m/s. Ia berkata "Benda berhenti." Lalu diam.'
    assert split_sentences(text) == [
        "Lihat Gb. 1.2 dan dll. Percepatan 9.8 m/s.",
        'Ia berkata "Benda berhenti."',
        "Lalu diam.",
    ]


def test_lowercase_after_period_is_not_a_boundary() -> None:
    assert split_sentences("Ini contoh dsb. dan seterusnya.") == ["Ini contoh dsb. dan seterusnya."]
