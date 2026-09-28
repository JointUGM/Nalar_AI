from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import (
    CHALLENGE_MOVES,
    MODEL_ANSWER_TYPES,
    MOVE_TABLE,
    AnswerType,
)


def test_move_table_follows_prd_7_2() -> None:
    assert MOVE_TABLE == {
        AnswerType.CORRECT_REASONED: (M.TRANSFER, M.DEEPER_REASON),
        AnswerType.CORRECT_UNREASONED: (M.REQUEST_JUSTIFICATION, M.DECOMPOSE),
        AnswerType.MISCONCEPTION: (M.COUNTER_EXAMPLE, M.EXPLAIN_MECHANISM, M.DECOMPOSE),
        AnswerType.MIXED: (M.COUNTER_EXAMPLE, M.REQUEST_JUSTIFICATION),
        AnswerType.EVASIVE: (M.DECOMPOSE, M.SIMPLER_REASON),
        AnswerType.MANIPULATION: (M.REFUSE_AND_REDIRECT,),
        AnswerType.UNSURE: (M.REQUEST_JUSTIFICATION,),
    }


def test_every_answer_type_except_safety_has_moves() -> None:
    assert set(MOVE_TABLE) == set(AnswerType) - {AnswerType.SAFETY}


def test_the_model_never_decides_unsure_or_safety() -> None:
    assert AnswerType.UNSURE not in MODEL_ANSWER_TYPES
    assert AnswerType.SAFETY not in MODEL_ANSWER_TYPES


def test_challenge_moves_are_counter_example_and_transfer() -> None:
    assert frozenset({M.COUNTER_EXAMPLE, M.TRANSFER}) == CHALLENGE_MOVES
