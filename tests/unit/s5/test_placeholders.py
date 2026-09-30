from nalar_ai.subsystems.s5_insight_synthesizer.domain.placeholders import (
    has_braces,
    rewrite,
    strip_placeholders,
)
from tests.support.s5 import C_GESEK, M_HABIS, make_counts

COUNTS = make_counts()
CONCEPTS, MISCONCEPTIONS = COUNTS.concept_aliases(), COUNTS.misconception_aliases()


def _rewrite(text: str) -> tuple[str, list[str]]:
    return rewrite(text, CONCEPTS, MISCONCEPTIONS)


def test_aliases_become_ids_and_class_totals_lose_their_prefix() -> None:
    text, problems = _rewrite(
        "{{count:m1}} dari {{class:total}}; {{mastered:c1}}; {{class:incomplete}}"
    )
    assert problems == []
    assert text == (
        f"{{{{count:{M_HABIS}}}}} dari {{{{total}}}}; {{{{mastered:{C_GESEK}}}}}; "
        "{{incomplete}}"
    )


def test_spacing_and_case_are_tolerated() -> None:
    text, problems = _rewrite("{{ Count : M1 }}")
    assert problems == [] and text == f"{{{{count:{M_HABIS}}}}}"


def test_an_unknown_alias_or_a_wrong_kind_is_rejected() -> None:
    assert _rewrite("{{count:m9}}")[1] == ["unknown placeholder '{{count:m9}}'"]
    assert _rewrite("{{count:c1}}")[1] == ["unknown placeholder '{{count:c1}}'"]
    assert _rewrite("{{total}}")[1] == ["unknown placeholder '{{total}}'"]


def test_stray_braces_are_rejected() -> None:
    assert _rewrite("ada {count:m1} siswa")[1] == ["write placeholders exactly as {{kind:id}}"]


def test_strip_and_braces_helpers() -> None:
    assert strip_placeholders("a {{count:m1}} b") == "a   b"
    assert has_braces("x {y}") and not has_braces("plain")
