import pytest

from nalar_ai.subsystems.s4_session_evaluator.domain.quotes import find_quote

ANSWER = "Lantainya menahan kelereng,  ada gesekan... jadi pelan-pelan berhenti."


def test_an_exact_quote_returns_itself() -> None:
    assert find_quote(ANSWER, "menahan kelereng") == "menahan kelereng"


@pytest.mark.parametrize(
    ("quote", "span"),
    [
        ("Menahan Kelereng", "menahan kelereng"),  # case
        ("kelereng, ada gesekan", "kelereng,  ada gesekan"),  # whitespace
        ("ada gesekan jadi pelan pelan", "ada gesekan... jadi pelan-pelan"),  # punctuation
    ],
)
def test_case_punctuation_and_spacing_are_ignored_and_the_original_span_returned(
    quote: str, span: str
) -> None:
    assert find_quote(ANSWER, quote) == span


@pytest.mark.parametrize(
    "quote",
    [
        "ada gaya gesek",  # words the student never wrote
        "gesekan ada",  # wrong order
        "lantai menahan",  # "lantai" is only part of the word "Lantainya"
        "",
        "...",
    ],
)
def test_invented_reordered_partial_or_empty_quotes_are_rejected(quote: str) -> None:
    assert find_quote(ANSWER, quote) is None


def test_a_one_word_answer_can_be_quoted() -> None:
    assert find_quote("gatau", "gatau") == "gatau"
