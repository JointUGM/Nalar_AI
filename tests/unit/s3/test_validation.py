from dataclasses import replace

import pytest

from nalar_ai.shared.enums import MoveReasonCode as R
from nalar_ai.shared.enums import PlannerMode
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import plan_move
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView
from nalar_ai.subsystems.s3_socratic_prober.domain.validation import (
    AcceptedChoice,
    ModelChoice,
    Rejection,
    validate_choice,
)
from tests.support.s3 import M_HABIS, T_LEMBAM, anchor, make_pack, question, question_id

PLAN = plan_move(
    Classification(AnswerType.MISCONCEPTION, ClassificationSource.MODEL, (M_HABIS,)),
    SessionView((anchor("gayanya habis", misconceptions=(M_HABIS,)),)),
    make_pack(),
    mode=PlannerMode.HYBRID,
)  # allowed: counter_example (default), explain_mechanism, decompose on T_LEMBAM

DEFAULT = ModelChoice(
    move=M.COUNTER_EXAMPLE,
    question_id=question_id(T_LEMBAM, M.COUNTER_EXAMPLE),
    reason_code=R.DEFAULT,
    reason="",
    question=" Kamu bilang gayanya habis. Kenapa pesawat luar angkasa tetap melaju? ",
)


def test_the_default_choice_is_accepted_and_trimmed() -> None:
    result = validate_choice(DEFAULT, PLAN)
    assert result == AcceptedChoice(
        move=M.COUNTER_EXAMPLE,
        question=question(T_LEMBAM, M.COUNTER_EXAMPLE),
        reason_code=R.DEFAULT,
        reason="",
        adapted_text="Kamu bilang gayanya habis. Kenapa pesawat luar angkasa tetap melaju?",
    )


def test_a_reasoned_departure_is_accepted() -> None:
    choice = replace(
        DEFAULT,
        move=M.DECOMPOSE,
        question_id=question_id(T_LEMBAM, M.DECOMPOSE),
        reason_code=R.FRUSTRATION,
        reason="Siswa terdengar bingung.",
    )
    result = validate_choice(choice, PLAN)
    assert isinstance(result, AcceptedChoice) and result.move is M.DECOMPOSE


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"move": M.TRANSFER}, "move transfer is not allowed"),
        ({"question_id": None}, "the question is not a candidate of move counter_example"),
        (
            {"question_id": question_id(T_LEMBAM, M.DECOMPOSE)},
            "the question is not a candidate of move counter_example",
        ),
        ({"reason_code": R.CHECK_DEEPER}, "the default move must use reason code default"),
        ({"question": "   "}, "the adapted question is empty"),
        (
            {"move": M.DECOMPOSE, "question_id": question_id(T_LEMBAM, M.DECOMPOSE)},
            "leaving the default move needs a reason code",
        ),
        (
            {
                "move": M.DECOMPOSE,
                "question_id": question_id(T_LEMBAM, M.DECOMPOSE),
                "reason_code": R.FRUSTRATION,
            },
            "leaving the default move needs a reason",
        ),
    ],
)
def test_invalid_choices_are_rejected(changes: dict[str, object], problem: str) -> None:
    assert validate_choice(replace(DEFAULT, **changes), PLAN) == Rejection(problem)  # type: ignore[arg-type]
