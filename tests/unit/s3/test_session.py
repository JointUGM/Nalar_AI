from dataclasses import replace

from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s3_socratic_prober.domain.session import (
    SessionView,
    TurnKind,
    validate_history,
)
from tests.support.s3 import M_HABIS, M_KASAR, T_GESEK, T_LEMBAM, anchor, make_pack, probe


def test_valid_history_has_no_problems() -> None:
    history = [anchor("karena gayanya habis"), probe(1, T_LEMBAM, M.COUNTER_EXAMPLE, "hmm")]
    assert validate_history(history, make_pack()) == []


def test_structure_is_strict() -> None:
    pack = make_pack()
    assert validate_history([], pack) == ["history is empty; it must start with the anchor turn"]
    bad = [
        replace(anchor("a"), kind=TurnKind.PROBE),
        replace(probe(1, T_GESEK, M.TRANSFER, "b"), turn_index=3, move=None),
        replace(probe(2, T_GESEK, M.TRANSFER, "c"), target_concept_id=M_HABIS),
    ]
    assert validate_history(bad, pack) == [
        "turn_index values must be 0, 1, 2, ... in order",
        "turn 0 must have kind 'anchor'",
        "probe turn 0 has no move",
        "probe turn 0 targets a concept outside the pack",
        "probe turn 1 has no move",
        "probe turn 2 targets a concept outside the pack",
    ]


def test_view_derives_coverage_and_asked_questions() -> None:
    view = SessionView(
        (
            anchor("a"),
            probe(1, T_GESEK, M.REQUEST_JUSTIFICATION, "b"),
            probe(2, T_GESEK, M.TRANSFER, "c"),
            probe(3, T_LEMBAM, M.DECOMPOSE, "d"),
        )
    )
    assert view.probes_asked == 3
    assert view.challenged == frozenset({T_GESEK})
    assert view.last_target == T_LEMBAM
    assert view.asked_question_ids == {
        "gesek-request_justification",
        "gesek-transfer",
        "lembam-decompose",
    }
    assert view.student_answers() == ("a", "b", "c", "d")


def test_trailing_decomposes_counts_only_the_tail() -> None:
    view = SessionView(
        (
            anchor("a"),
            probe(1, T_GESEK, M.DECOMPOSE, "b"),
            probe(2, T_GESEK, M.TRANSFER, "c"),
            probe(3, T_GESEK, M.DECOMPOSE, "d"),
            probe(4, T_GESEK, M.DECOMPOSE, "e"),
        )
    )
    assert view.trailing_decomposes == 2
    assert SessionView((anchor("a"),)).trailing_decomposes == 0


def test_counter_streak_counts_unchanged_counter_examples() -> None:
    habis = (M_HABIS,)
    view = SessionView(
        (
            anchor("gayanya habis", misconceptions=habis),
            probe(1, T_LEMBAM, M.COUNTER_EXAMPLE, "tetap habis", misconceptions=habis),
            probe(2, T_LEMBAM, M.COUNTER_EXAMPLE, "masih habis kok"),
        )
    )
    assert view.counter_streak(M_HABIS, habis) == 2  # the latest answer still shows it
    assert view.counter_streak(M_HABIS, ()) == 0  # the latest answer changed
    assert view.counter_streak(M_KASAR, (M_KASAR,)) == 0


def test_counter_streak_stops_at_another_move() -> None:
    habis = (M_HABIS,)
    view = SessionView(
        (
            anchor("habis", misconceptions=habis),
            probe(1, T_LEMBAM, M.COUNTER_EXAMPLE, "habis", misconceptions=habis),
            probe(2, T_LEMBAM, M.EXPLAIN_MECHANISM, "habis", misconceptions=habis),
            probe(3, T_LEMBAM, M.COUNTER_EXAMPLE, "habis"),
        )
    )
    assert view.counter_streak(M_HABIS, habis) == 1
