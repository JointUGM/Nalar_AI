from dataclasses import replace
from pathlib import Path

import pytest

from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s4_session_evaluator.domain.reflection import (
    ReflectionLexicon,
    ReflectionParts,
    check_reflection,
)
from nalar_ai.subsystems.s4_session_evaluator.infrastructure.config_loader import (
    DEFAULT_CONFIG_PATH,
    load_evaluator_config,
)

LEXICON = ReflectionLexicon.build(
    verdict_terms=["benar", "Tepat sekali", "bagus"],
    banned_terms=["nilai", "menyontek"],
    max_words=40,
)
TERMS = ("gesekan", "kelembaman")
ANSWERS = ("Lantainya menahan kelereng, ada gesekan jadi pelan-pelan berhenti.",)
GOOD = ReflectionParts(
    strengths="Kamu memakai contoh lantai untuk menjelaskan pikiranmu.",
    changed_mind="Awalnya kamu bilang gayanya habis, lalu kamu memikirkannya lagi.",
    question="Apa yang akan kamu perhatikan kalau kelereng digelindingkan di karpet?",
)


def _check(parts: ReflectionParts, answers: tuple[str, ...] = ANSWERS) -> list[str]:
    return check_reflection(parts, answers, TERMS, LEXICON)


def test_a_friendly_reflection_passes() -> None:
    assert _check(GOOD) == []
    assert GOOD.content.count("\n\n") == 2


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"question": "Apa yang kamu perhatikan."}, "the question must end with '?'"),
        ({"strengths": "  "}, "every part must be written"),
        ({"strengths": "Kamu menjawab 3 pertanyaan."}, "write no digits or numbers"),
        ({"strengths": "Jawabanmu sudah benar."}, "do not judge right or wrong (found 'benar')"),
        (
            {"strengths": "Tepat sekali, kamu hebat."},
            "do not judge right or wrong (found 'tepat sekali')",
        ),
        (
            {"strengths": "Nilaimu naik."},
            "do not mention scores, cheating or copying (found 'nilai')",
        ),
        (
            {"changed_mind": "Ini soal kelembamannya benda."},
            "do not name a science idea or term the student did not write",
        ),
        ({"strengths": "kata " * 40}, "use at most 40 words in total"),
    ],
)
def test_each_rule_blocks_its_case(change: dict[str, str], problem: str) -> None:
    assert problem in _check(replace(GOOD, **change))


def test_the_students_own_science_words_may_be_echoed() -> None:
    echo = replace(GOOD, strengths="Kamu menyebut gesekannya lantai sebagai alasan.")
    assert _check(echo) == []
    assert _check(echo, answers=("gayanya habis",)) == [
        "do not name a science idea or term the student did not write"
    ]


def test_the_shipped_config_loads_and_its_template_passes_the_guard() -> None:
    config = load_evaluator_config()
    assert config.version == 1
    assert "bagus" in config.lexicon.verdict_terms
    assert check_reflection(config.template, (), (), config.lexicon) == []


def test_a_template_that_fails_the_guard_is_refused_at_startup(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "ingin kamu cari tahu lebih jauh?", "ingin kamu cari tahu lebih jauh."
    )
    path = tmp_path / "evaluator.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="template fails the guard"):
        load_evaluator_config(path)


def test_a_term_hidden_inside_another_student_word_is_not_an_exemption() -> None:
    # "harus" contains "arus", but the student never wrote the idea "arus".
    leak = replace(GOOD, strengths="Kamu bilang arus listrik mengalir ke lampu.")
    problems = check_reflection(leak, ("lampunya harus nyala",), ("arus",), LEXICON)
    assert problems == ["do not name a science idea or term the student did not write"]


def test_the_students_own_affixed_form_may_be_echoed_but_not_the_bare_term() -> None:
    answers = ("kelerengnya bergesekan dengan lantai",)
    same_form = replace(GOOD, strengths="Kamu bilang kelerengnya bergesekan dengan lantai.")
    bare_term = replace(GOOD, strengths="Kamu menyebut gesekan sebagai alasannya.")
    assert check_reflection(same_form, answers, TERMS, LEXICON) == []
    assert check_reflection(bare_term, answers, TERMS, LEXICON) == [
        "do not name a science idea or term the student did not write"
    ]
