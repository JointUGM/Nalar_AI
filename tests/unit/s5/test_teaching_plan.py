from dataclasses import replace

import pytest

from nalar_ai.subsystems.s5_insight_synthesizer.domain.teaching_plan import (
    PlanStep,
    TeachingPlan,
    check_plan,
)
from tests.support.s5 import CONFIG

PLAN = TeachingPlan(
    objective="Siswa memahami bahwa benda berat dan ringan jatuh hampir bersamaan tanpa hambatan udara.",
    steps=(
        PlanStep(
            "elicit", "Pegang bola dan kertas kusut setinggi bahu.", "Mana yang sampai lebih dulu?"
        ),
        PlanStep("contrast", "Jatuhkan keduanya bersamaan.", "Apa yang kalian lihat?"),
        PlanStep(
            "explain", "Sebut gagasan keliru lalu jelaskan.", "Jelaskan lagi dengan kata sendiri."
        ),
    ),
    counter_example=1,
    check_question="Jika batu besar dan kecil dijatuhkan, mana yang sampai dulu?",
    look_for="Siswa menyebut hambatan udara, bukan berat, sebagai penyebab perbedaan.",
    question_ids=("q1",),
)


def _check(plan: TeachingPlan, examples: int = 1) -> list[str]:
    return check_plan(
        plan,
        counter_examples=examples,
        question_ids=frozenset({"q1"}),
        lexicon=CONFIG.numbers,
        limits=CONFIG.teaching_plan,
    )


def test_a_grounded_plan_passes_and_digits_are_allowed() -> None:
    assert _check(replace(PLAN, check_question="Bola 2 kg dan 1 kg dijatuhkan, mana duluan?")) == []


@pytest.mark.parametrize(
    ("plan", "examples", "expected"),
    [
        (replace(PLAN, steps=PLAN.steps[:2]), 1, "steps must be exactly"),
        (replace(PLAN, objective="Sebagian besar siswa masih keliru."), 1, "how many students"),
        (replace(PLAN, counter_example=None), 1, "counter_example: choose"),
        (replace(PLAN, counter_example=2), 1, "counter_example: use only"),
        (replace(PLAN, counter_example=1), 0, "counter_example: use only"),
        (replace(PLAN, question_ids=("q9",)), 1, "question_ids"),
        (replace(PLAN, look_for=" "), 1, "look_for: must not be empty"),
        (replace(PLAN, objective="kata " * 31), 1, "at most 30 words"),
    ],
)
def test_guard_problems(plan: TeachingPlan, examples: int, expected: str) -> None:
    assert any(expected in problem for problem in _check(plan, examples))
