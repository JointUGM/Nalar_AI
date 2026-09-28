import pytest

from nalar_ai.shared.enums import GuardResult as G
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import (
    GuardLexicon,
    GuardThresholds,
    check_similarity,
    check_text,
    count_sentences,
)

LEXICON = GuardLexicon.build(
    verdict_terms=["benar", "salah", "tepat", "bagus", "yang benar adalah"],
    max_chars=280,
    max_sentences=2,
)
APPROVED = "Kalau dorongannya habis, kenapa pesawat luar angkasa tetap melaju walau mesinnya mati?"
TERMS = ("gesekan", "kelembaman", "inersia", "hukum newton")
THRESHOLDS = GuardThresholds(max_reference_similarity=0.8, min_approved_similarity=0.55)


def _check(adapted: str, answers: tuple[str, ...] = ()) -> G | None:
    return check_text(adapted, APPROVED, answers, TERMS, LEXICON)


def test_a_light_adaptation_passes() -> None:
    assert _check("Kamu bilang dorongannya habis. Kenapa pesawat itu tetap melaju?") is None


@pytest.mark.parametrize(
    "adapted",
    [
        "",
        "Kenapa pesawat itu tetap melaju.",  # no question mark
        "Oke. Kamu bilang habis. Kenapa pesawat itu tetap melaju?",  # three sentences
        "Kenapa " + "sangat " * 60 + "melaju?",  # too long
    ],
)
def test_shape(adapted: str) -> None:
    assert _check(adapted) is G.BLOCKED_SHAPE


def test_verdicts_are_blocked_unless_the_approved_question_has_them() -> None:
    assert _check("Jawabanmu hampir benar. Kenapa pesawat tetap melaju?") is G.BLOCKED_VERDICT
    approved = "Apakah benar dorongannya habis?"
    assert check_text("Menurutmu, apakah benar begitu?", approved, (), TERMS, LEXICON) is None


def test_new_answer_terms_are_blocked_unless_the_student_said_them() -> None:
    adapted = "Apa hubungan gesekan dengan pesawat yang tetap melaju?"
    assert _check(adapted) is G.BLOCKED_NEW_TERMS
    assert _check(adapted, answers=("mungkin karena Gesekan?",)) is None
    assert _check("Kenapa kelembamannya begitu?") is G.BLOCKED_NEW_TERMS  # affixed forms count


def test_sentence_count() -> None:
    assert count_sentences("Kenapa?") == 1
    assert count_sentences("Kamu bilang habis. Kenapa pesawat tetap melaju?") == 2
    assert count_sentences("  ") == 0


def test_similarity_blocks_text_closer_to_the_reference_than_the_approved_one() -> None:
    reference = (1.0, 0.0, 0.0)
    approved = (0.6, 0.8, 0.0)
    close = (0.9, 0.436, 0.0)  # cos to reference 0.9, cos to approved about 0.89
    assert check_similarity(close, approved, reference, THRESHOLDS) is G.BLOCKED_SIMILARITY
    # when the approved question is itself that close, the adaptation is no worse
    assert check_similarity(close, close, reference, THRESHOLDS) is G.PASSED


def test_similarity_blocks_drift_from_the_approved_question() -> None:
    reference = (1.0, 0.0, 0.0)
    approved = (0.0, 1.0, 0.0)
    drifted = (0.0, 0.3, 0.954)
    assert check_similarity(drifted, approved, reference, THRESHOLDS) is G.BLOCKED_DRIFT
    assert check_similarity(approved, approved, reference, THRESHOLDS) is G.PASSED


@pytest.mark.parametrize(
    "adapted",
    [
        "Kalau lantainya tidak ada gesekannya, kenapa kelereng tetap berhenti?",
        "Apakah kelereng itu bergesekan dengan lantai?",
        "Karena inersianya, kenapa pesawat tetap melaju?",
    ],
)
def test_affixed_answer_terms_are_blocked(adapted: str) -> None:
    assert _check(adapted) is G.BLOCKED_NEW_TERMS


def test_a_term_the_student_used_with_an_affix_is_known() -> None:
    assert (
        _check(
            "Kamu bilang gesekannya kecil. Kenapa kelereng tetap berhenti?", ("gesekannya kecil",)
        )
        is None
    )


@pytest.mark.parametrize(
    ("term", "answer", "adapted"),
    [
        ("arus", "lampunya harus nyala", "Kalau arus listriknya putus, kenapa lampu padam?"),
        ("sel", "lampunya selalu nyala", "Apa yang terjadi di dalam sel itu?"),
        ("gas", "tugasnya susah", "Kenapa gas itu mengembang?"),
    ],
)
def test_a_term_hidden_inside_another_student_word_is_not_known(
    term: str, answer: str, adapted: str
) -> None:
    # "harus" contains "arus", but the student never named the idea "arus".
    assert check_text(adapted, APPROVED, (answer,), (term,), LEXICON) is G.BLOCKED_NEW_TERMS


def test_only_the_students_exact_affixed_form_is_known_not_the_bare_term() -> None:
    answers = ("kelerengnya bergesekan dengan lantai",)
    same_form = "Kamu bilang kelerengnya bergesekan. Kenapa tetap berhenti?"
    bare_term = "Kamu bilang ada gesekan. Kenapa tetap berhenti?"
    assert _check(same_form, answers) is None
    assert _check(bare_term, answers) is G.BLOCKED_NEW_TERMS


def test_known_terms_never_match_across_two_sources() -> None:
    approved = "Menurutmu, apa arti kata hukum?"
    adapted = "Menurutmu, apa arti hukum newton?"
    assert (
        check_text(adapted, approved, ("newton itu ilmuwan",), TERMS, LEXICON)
        is G.BLOCKED_NEW_TERMS
    )
