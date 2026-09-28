from dataclasses import replace
from uuid import UUID

import pytest

from nalar_ai.shared.enums import PlannerMode
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import (
    EndReason,
    check_end,
    coverage,
    plan_move,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType as T
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import BankQuestion
from nalar_ai.subsystems.s3_socratic_prober.domain.session import HistoryTurn, SessionView
from tests.support.s3 import (
    M_HABIS,
    M_KASAR,
    T_GESEK,
    T_LEMBAM,
    anchor,
    make_pack,
    probe,
    question_id,
)

HYBRID = PlannerMode.HYBRID


def _c(answer_type: T, *ids: UUID) -> Classification:
    return Classification(answer_type, ClassificationSource.MODEL, ids)


def _view(*turns: HistoryTurn) -> SessionView:
    return SessionView((anchor("jawaban awal"), *turns))


def test_first_probe_after_a_bare_claim_asks_why_about_the_first_target() -> None:
    plan = plan_move(_c(T.CORRECT_UNREASONED), _view(), make_pack(), mode=HYBRID)
    assert plan.target_concept_id == T_GESEK
    assert plan.allowed_moves == (M.REQUEST_JUSTIFICATION, M.DECOMPOSE)
    assert plan.default_question.id == question_id(T_GESEK, M.REQUEST_JUSTIFICATION)
    assert not plan.coverage_forced


def test_a_misconception_targets_its_concept_with_its_counter_example_first() -> None:
    plan = plan_move(_c(T.MISCONCEPTION, M_HABIS), _view(), make_pack(), mode=HYBRID)
    assert plan.target_concept_id == T_LEMBAM
    assert plan.allowed_moves == (M.COUNTER_EXAMPLE, M.EXPLAIN_MECHANISM, M.DECOMPOSE)
    assert plan.default_question.misconception_id == M_HABIS


def test_matching_counter_examples_come_before_others() -> None:
    pack = make_pack()
    generic = BankQuestion("lembam-counter-2", T_LEMBAM, M.COUNTER_EXAMPLE, "Kenapa bola di es?")
    pack = replace(pack, question_bank=(generic, *pack.question_bank))
    plan = plan_move(_c(T.MISCONCEPTION, M_HABIS), _view(), pack, mode=HYBRID)
    assert [q.id for q in plan.candidates[M.COUNTER_EXAMPLE]] == [
        question_id(T_LEMBAM, M.COUNTER_EXAMPLE),
        "lembam-counter-2",
    ]


def test_asked_questions_are_skipped_unless_nothing_else_is_left() -> None:
    pack = make_pack()
    extra = BankQuestion("gesek-deeper-2", T_GESEK, M.DEEPER_REASON, "Bagaimana di atas pasir?")
    pack = replace(pack, question_bank=(*pack.question_bank, extra))
    view = _view(probe(1, T_GESEK, M.DEEPER_REASON, "b"))
    plan = plan_move(_c(T.CORRECT_REASONED), view, pack, mode=HYBRID)
    assert [q.id for q in plan.candidates[M.DEEPER_REASON]] == ["gesek-deeper-2"]
    # the sample pack has one deeper_reason question, so it stays available after being asked
    plan = plan_move(_c(T.CORRECT_REASONED), view, make_pack(), mode=HYBRID)
    assert [q.id for q in plan.candidates[M.DEEPER_REASON]] == [
        question_id(T_GESEK, M.DEEPER_REASON)
    ]


def test_table_mode_allows_only_the_default() -> None:
    plan = plan_move(_c(T.MISCONCEPTION, M_HABIS), _view(), make_pack(), mode=PlannerMode.TABLE)
    assert plan.allowed_moves == (M.COUNTER_EXAMPLE,)


def test_moves_on_after_the_target_was_challenged() -> None:
    view = _view(probe(1, T_GESEK, M.TRANSFER, "di es lebih jauh karena licin"))
    plan = plan_move(_c(T.CORRECT_REASONED), view, make_pack(), mode=HYBRID)
    assert plan.target_concept_id == T_LEMBAM


def test_stays_on_an_unchallenged_target() -> None:
    view = _view(probe(1, T_GESEK, M.REQUEST_JUSTIFICATION, "karena lantai"))
    plan = plan_move(_c(T.CORRECT_UNREASONED), view, make_pack(), mode=HYBRID)
    assert plan.target_concept_id == T_GESEK


def test_coverage_forcing_overrides_a_manipulation_answer() -> None:
    # 5 probes allowed, 3 used, nothing challenged: 2 left for 2 targets.
    view = _view(
        probe(1, T_GESEK, M.REQUEST_JUSTIFICATION, "a"),
        probe(2, T_GESEK, M.DECOMPOSE, "b"),
        probe(3, T_LEMBAM, M.REFUSE_AND_REDIRECT, "c"),
    )
    plan = plan_move(_c(T.MANIPULATION), view, make_pack(), mode=HYBRID)
    assert plan.coverage_forced
    assert plan.target_concept_id == T_LEMBAM  # the last target, still unchallenged
    assert plan.allowed_moves == (M.TRANSFER, M.COUNTER_EXAMPLE)


def test_coverage_forcing_counters_a_misconception_of_the_target() -> None:
    view = _view(
        probe(1, T_GESEK, M.TRANSFER, "a"),
        probe(2, T_GESEK, M.REQUEST_JUSTIFICATION, "b"),
        probe(3, T_GESEK, M.DECOMPOSE, "c"),
        probe(4, T_GESEK, M.DECOMPOSE, "d"),
    )
    plan = plan_move(_c(T.MISCONCEPTION, M_HABIS), view, make_pack(), mode=HYBRID)
    assert plan.coverage_forced and plan.target_concept_id == T_LEMBAM
    assert plan.allowed_moves == (M.COUNTER_EXAMPLE, M.TRANSFER)


def test_forcing_prefers_transfer_when_the_misconception_is_elsewhere() -> None:
    view = _view(
        probe(1, T_GESEK, M.TRANSFER, "a"),
        probe(2, T_GESEK, M.REQUEST_JUSTIFICATION, "b"),
        probe(3, T_GESEK, M.DECOMPOSE, "c"),
        probe(4, T_GESEK, M.DECOMPOSE, "d"),
    )
    plan = plan_move(_c(T.MISCONCEPTION, M_KASAR), view, make_pack(), mode=HYBRID)
    assert plan.target_concept_id == T_LEMBAM
    assert plan.allowed_moves == (M.TRANSFER, M.COUNTER_EXAMPLE)


def test_a_misconception_countered_twice_without_change_is_not_countered_again() -> None:
    habis = (M_HABIS,)
    view = SessionView(
        (
            anchor("gayanya habis", misconceptions=habis),
            probe(1, T_LEMBAM, M.COUNTER_EXAMPLE, "tetap habis", misconceptions=habis),
            probe(2, T_LEMBAM, M.COUNTER_EXAMPLE, "masih habis"),
        )
    )
    pack = replace(make_pack(), max_probes=6)
    plan = plan_move(_c(T.MISCONCEPTION, M_HABIS), view, pack, mode=HYBRID)
    assert plan.allowed_moves == (M.EXPLAIN_MECHANISM, M.DECOMPOSE)


def test_no_third_decompose_in_a_row() -> None:
    view = _view(
        probe(1, T_GESEK, M.DECOMPOSE, "gatau"),
        probe(2, T_GESEK, M.DECOMPOSE, "gatau"),
    )
    pack = replace(make_pack(), max_probes=6)
    plan = plan_move(_c(T.EVASIVE), view, pack, mode=HYBRID)
    assert plan.allowed_moves == (M.SIMPLER_REASON,)


def test_safety_has_no_move() -> None:
    with pytest.raises(ValueError, match="safety"):
        plan_move(_c(T.SAFETY), _view(), make_pack(), mode=HYBRID)


def test_candidate_lookup_checks_the_move() -> None:
    plan = plan_move(_c(T.CORRECT_UNREASONED), _view(), make_pack(), mode=HYBRID)
    justify = question_id(T_GESEK, M.REQUEST_JUSTIFICATION)
    assert plan.candidate(M.REQUEST_JUSTIFICATION, justify) is not None
    assert plan.candidate(M.DECOMPOSE, justify) is None
    assert plan.candidate(M.TRANSFER, justify) is None


def test_end_rules() -> None:
    pack = make_pack()  # 5 probes, 15 minutes
    assert check_end(_view(), pack, min_probes=4, elapsed_seconds=900) is EndReason.TIME_LIMIT
    five = _view(*(probe(i, T_GESEK, M.DECOMPOSE, "x") for i in range(1, 6)))
    assert check_end(five, pack, min_probes=4, elapsed_seconds=10) is EndReason.TURN_LIMIT
    covered = _view(
        probe(1, T_GESEK, M.TRANSFER, "a"),
        probe(2, T_LEMBAM, M.COUNTER_EXAMPLE, "b"),
    )
    assert check_end(covered, pack, min_probes=4, elapsed_seconds=10) is None
    assert check_end(covered, pack, min_probes=2, elapsed_seconds=10) is (
        EndReason.COVERAGE_COMPLETE
    )
    short = replace(pack, max_probes=2)
    assert check_end(covered, short, min_probes=4, elapsed_seconds=10) is EndReason.TURN_LIMIT


def test_coverage_lists_every_target_in_pack_order() -> None:
    assert coverage(make_pack(), frozenset({T_LEMBAM})) == ((T_GESEK, False), (T_LEMBAM, True))
