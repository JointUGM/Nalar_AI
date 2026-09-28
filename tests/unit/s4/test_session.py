from dataclasses import replace
from uuid import UUID

from nalar_ai.shared.enums import RubricDimension
from nalar_ai.subsystems.s4_session_evaluator.domain.session import (
    Rubric,
    TurnKind,
    validate_session,
)
from tests.support.s4 import TURN_IDS, make_rubric, make_session, make_turns


def test_the_sample_session_is_valid() -> None:
    session = make_session()
    assert validate_session(session) == []
    assert [t.alias for t in session.answered_turns] == ["t0", "t1", "t2", "t3", "t4"]


def test_an_unanswered_last_turn_is_not_an_answered_turn() -> None:
    session = make_session(turns=make_turns(["a", "b", "  "]))
    assert [t.alias for t in session.answered_turns] == ["t0", "t1"]
    assert validate_session(session) == []


def test_a_session_without_any_answer_is_rejected() -> None:
    errors = validate_session(make_session(turns=make_turns([""])))
    assert errors == ["turns: no answered turn to evaluate"]


def test_turn_structure_is_checked() -> None:
    turns = list(make_turns())
    turns[2] = replace(turns[2], turn_id=TURN_IDS[1])
    turns[3] = replace(turns[3], turn_index=7)
    turns[4] = replace(turns[4], kind=TurnKind.ANCHOR)
    assert validate_session(make_session(turns=tuple(turns))) == [
        "turns: duplicate turn_id",
        "turns: turn_index must run 0, 1, 2, ... in order",
        "turn 4: only turn 0 is the anchor",
    ]


def test_a_probe_needs_a_known_target_and_a_move() -> None:
    turns = list(make_turns())
    turns[1] = replace(turns[1], target_concept_id=UUID(int=0x999))
    turns[2] = replace(turns[2], move=None)
    assert validate_session(make_session(turns=tuple(turns))) == [
        "turn 1: target is not in the context pack",
        "turn 2: a probe needs a move and a target",
    ]


def test_every_rubric_dimension_needs_five_descriptors() -> None:
    descriptors = dict(make_rubric().descriptors)
    descriptors[RubricDimension.TRANSFER] = descriptors[RubricDimension.TRANSFER][:4]
    del descriptors[RubricDimension.CLAIM]
    assert validate_session(make_session(rubric=Rubric(descriptors))) == [
        "rubric.claim: needs exactly 5 descriptors",
        "rubric.transfer: needs exactly 5 descriptors",
    ]


def test_misconceptions_must_belong_to_a_target() -> None:
    session = make_session()
    stray = replace(session.pack.misconceptions[0], concept_id=UUID(int=0x999))
    pack = replace(session.pack, misconceptions=(stray, session.pack.misconceptions[1]))
    (error,) = validate_session(replace(session, pack=pack))
    assert error.endswith("is not a target")


def test_a_session_whose_answers_have_no_words_is_rejected_before_scoring() -> None:
    # "..." counts as answered (turn quality 0), but nothing in it can ever be quoted.
    errors = validate_session(make_session(turns=make_turns(["...", "?", "🤷"])))
    assert errors == ["turns: no answer contains words to quote"]
    assert validate_session(make_session(turns=make_turns(["...", "gatau"]))) == []
