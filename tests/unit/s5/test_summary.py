import pytest

from nalar_ai.shared.enums import ConceptOutcome as O
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    ParentConcept,
    SummaryRules,
    SummaryTemplate,
    check_summary,
)
from nalar_ai.subsystems.s5_insight_synthesizer.infrastructure.config_loader import (
    load_synthesizer_config,
)
from tests.support.s5 import SUMMARY, make_parent_input

RULES = SummaryRules.build(
    min_words=60,
    max_words=150,
    banned_terms=["nilai", "nyontek"],
    comparison_phrases=["teman sekelas", "rata-rata"],
)
TEMPLATE = SummaryTemplate(
    opening="Ananda telah menyelesaikan misi penalaran sains.",
    understood="Ananda sudah memahami {concepts}.",
    developing="Ananda masih mengembangkan pemahaman tentang {concepts}.",
    home="Di rumah, ajak Ananda bercerita tentang {concept}.",
)


def test_a_clean_summary_passes() -> None:
    assert check_summary(SUMMARY, RULES) == []


@pytest.mark.parametrize("word", ["saling", "SALING!"])
def test_shipped_guard_accepts_the_word_saling(word: str) -> None:
    text = SUMMARY + f" Gaya dan gerak {word} berkaitan dalam kegiatan sehari-hari."
    assert check_summary(text, load_synthesizer_config().summary) == []


@pytest.mark.parametrize(
    "word", ["salin", "menyalin", "disalin", "menyalinnya", "saling menyalin", "salingnya"]
)
def test_shipped_guard_still_rejects_copying_and_unapproved_affixes(word: str) -> None:
    problems = check_summary(
        SUMMARY + f" Ananda {word} jawaban.", load_synthesizer_config().summary
    )
    assert any(
        problem.startswith("do not mention scores, cheating or copying") for problem in problems
    )


def test_exceptions_are_normalized_exact_and_scoped_to_their_banned_term() -> None:
    rules = SummaryRules.build(
        min_words=1,
        max_words=20,
        banned_terms=["salin", "saling"],
        comparison_phrases=[],
        banned_term_exceptions={"SALIN!": ["SALING!", "saling", ""]},
    )
    assert rules.banned_term_exceptions == (("salin", ("saling",)),)
    # The exception for "salin" cannot override a separately banned term.
    assert check_summary("saling", rules) == [
        "do not mention scores, cheating or copying (found 'saling')"
    ]
    assert "do not mention scores, cheating or copying (found 'salin')" in check_summary(
        "salingnya", rules
    )


def test_a_provided_numeric_name_does_not_exempt_a_longer_number() -> None:
    assert check_summary(SUMMARY + " Misi Hukum Newton 11 selesai.", RULES, ["Hukum Newton 1"]) == [
        "write no digits or numbers"
    ]


def test_length_digits_scores_and_comparisons_are_rejected() -> None:
    assert check_summary("Terlalu pendek.", RULES) == ["write at least 60 words"]
    assert check_summary("kata " * 151, RULES) == ["use at most 150 words"]
    text = SUMMARY + " Nilainya 3, di atas rata-rata teman sekelas; Ananda tidak menyontek."
    assert check_summary(text, RULES) == [
        "write no digits or numbers",
        "do not mention scores, cheating or copying (found 'nilai')",
        "do not mention scores, cheating or copying (found 'nyontek')",
        "do not compare Ananda with other students (found 'teman sekelas')",
        "do not compare Ananda with other students (found 'rata rata')",
    ]


def test_a_concept_name_with_a_digit_is_exempt() -> None:
    names = make_parent_input(mission_title="Hukum Newton 1").names()
    assert check_summary(SUMMARY + " Misi Hukum Newton 1 selesai.", RULES, names) == []


def test_the_template_names_understood_and_developing_concepts() -> None:
    assert TEMPLATE.render(make_parent_input().concepts) == (
        "Ananda telah menyelesaikan misi penalaran sains. Ananda sudah memahami Gaya gesek. "
        "Ananda masih mengembangkan pemahaman tentang Kelembaman. "
        "Di rumah, ajak Ananda bercerita tentang Kelembaman."
    )


def test_the_template_reads_well_when_everything_or_nothing_is_mastered() -> None:
    all_mastered = [ParentConcept(n, O.MASTERED, None, False) for n in ("A", "B", "C")]
    assert TEMPLATE.render(all_mastered) == (
        "Ananda telah menyelesaikan misi penalaran sains. Ananda sudah memahami A, B dan C. "
        "Di rumah, ajak Ananda bercerita tentang A."
    )
    none = [
        ParentConcept("A", O.NOT_OBSERVED, None, False),
        ParentConcept("B", O.DEVELOPING, None, False),
    ]
    assert TEMPLATE.render(none) == (
        "Ananda telah menyelesaikan misi penalaran sains. "
        "Ananda masih mengembangkan pemahaman tentang A dan B. "
        "Di rumah, ajak Ananda bercerita tentang A."
    )
