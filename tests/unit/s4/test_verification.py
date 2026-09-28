from collections.abc import Callable
from typing import Any

import pytest

from nalar_ai.shared.enums import ConceptOutcome, RubricDimension
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import (
    Evaluation,
    VerificationLimits,
    verify,
)
from tests.support.s3 import M_HABIS, M_KASAR, T_GESEK, T_LEMBAM
from tests.support.s4 import ANSWERS, TURN_IDS, draft, make_session, make_turns, scoring_payload

LIMITS = VerificationLimits()


def _verify(mutate: Callable[[dict[str, Any]], None]) -> list[str]:
    payload = scoring_payload()
    mutate(payload)
    evaluation, errors = verify(draft(payload), make_session(), LIMITS)
    assert (evaluation is None) == bool(errors)
    return errors


def test_a_valid_answer_resolves_every_alias_to_ids() -> None:
    evaluation, errors = verify(draft(), make_session(), LIMITS)
    assert errors == []
    assert isinstance(evaluation, Evaluation)
    assert [s.dimension for s in evaluation.scores] == list(RubricDimension)
    assert evaluation.scores[0].evidence[0].turn_id == TURN_IDS[2]
    lembam = evaluation.concept_results[1]
    assert lembam.concept_id == T_LEMBAM
    assert lembam.outcome is ConceptOutcome.DEVELOPING
    assert lembam.initial_misconception_id == M_HABIS
    assert lembam.resolved_in_session is True
    assert lembam.evidence_turn_id == TURN_IDS[1]
    assert [(q.turn_index, q.quality) for q in evaluation.turn_quality] == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 3),
        (4, 0),
    ]


def test_the_stored_quote_is_the_students_own_span() -> None:
    def tidy(p: dict[str, Any]) -> None:
        p["scores"][1]["evidence"] = [{"turn": "t2", "quote": "lantainya MENAHAN kelereng"}]

    payload = scoring_payload()
    tidy(payload)
    evaluation, _ = verify(draft(payload), make_session(), LIMITS)
    assert evaluation is not None
    assert evaluation.scores[1].evidence[0].quote == "Lantainya menahan kelereng"


def test_a_missing_or_repeated_dimension_is_rejected() -> None:
    def repeat(p: dict[str, Any]) -> None:
        p["scores"][3]["dimension"] = "claim"

    assert _verify(repeat)[0] == (
        "scores: list claim, evidence, mechanism and transfer exactly once each"
    )


def test_levels_must_be_zero_to_four() -> None:
    def too_high(p: dict[str, Any]) -> None:
        p["scores"][0]["level"] = 5

    assert _verify(too_high) == ["scores[claim].level: must be 0 to 4"]


def test_every_score_needs_evidence_even_at_level_zero() -> None:
    def empty(p: dict[str, Any]) -> None:
        p["scores"][3].update(level=0, evidence=[])

    (error,) = _verify(empty)
    assert error.startswith("scores[transfer].evidence: give 1 to 4 quotes")


def test_a_level_zero_may_quote_a_one_word_attempt() -> None:
    def gatau(p: dict[str, Any]) -> None:
        p["scores"][3].update(level=0, evidence=[{"turn": "t4", "quote": "gatau"}])

    assert _verify(gatau) == []


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (
            {"turn": "t2", "quote": "ada gaya gesek yang melawan"},
            "scores[claim].evidence[0].quote: not found in the student's answer in t2",
        ),
        (
            {"turn": "t2", "quote": "Apa yang membuatmu yakin?"},  # the question, not the answer
            "scores[claim].evidence[0].quote: not found in the student's answer in t2",
        ),
        (
            {"turn": "t9", "quote": "gatau"},
            "scores[claim].evidence[0].turn: 't9' is not an answered",
        ),
    ],
)
def test_quotes_must_come_from_the_cited_answer(evidence: dict[str, str], message: str) -> None:
    def invent(p: dict[str, Any]) -> None:
        p["scores"][0]["evidence"] = [evidence]

    (error,) = _verify(invent)
    assert error.startswith(message)


def test_an_unanswered_turn_cannot_be_quoted_or_rated() -> None:
    session = make_session(turns=make_turns([*ANSWERS[:4], ""]))
    payload = scoring_payload()
    payload["scores"][3]["evidence"] = [{"turn": "t4", "quote": "gatau"}]
    _, errors = verify(draft(payload), session, LIMITS)
    assert errors == [
        "scores[transfer].evidence[0].turn: 't4' is not an answered turn",
        "turn_quality: rate exactly these turns once each: t0, t1, t2, t3",
    ]


def test_concepts_need_exactly_one_entry_per_target() -> None:
    def drop(p: dict[str, Any]) -> None:
        del p["concepts"][1]

    assert _verify(drop) == ["concepts: give exactly one entry for each of c1, c2"]


def test_a_misconception_outcome_must_name_one_of_that_concepts_wrong_ideas() -> None:
    def unnamed(p: dict[str, Any]) -> None:
        p["concepts"][1].update(outcome="misconception")

    def other_concept(p: dict[str, Any]) -> None:
        p["concepts"][1].update(outcome="misconception", misconception="m2")

    assert _verify(unnamed) == [
        "concepts[c2]: name a misconception exactly when the outcome is 'misconception'"
    ]
    assert _verify(other_concept) == ["concepts[c2].misconception: m2 belongs to another concept"]


@pytest.mark.parametrize(
    ("initial", "outcome", "misconception", "resolved"),
    [
        ("m1", "developing", None, True),  # started with the wrong idea, no longer holds it
        ("m1", "mastered", None, True),
        ("m1", "misconception", "m1", False),  # still holds it
        (None, "developing", None, False),  # never held one
    ],
)
def test_resolved_in_session_is_computed_by_code_not_taken_from_the_model(
    initial: str | None, outcome: str, misconception: str | None, resolved: bool
) -> None:
    payload = scoring_payload()
    payload["concepts"][1].update(
        initial_misconception=initial, outcome=outcome, misconception=misconception
    )
    evaluation, errors = verify(draft(payload), make_session(), LIMITS)
    assert errors == [] and evaluation is not None
    assert evaluation.concept_results[1].resolved_in_session is resolved


def test_not_observed_cannot_start_with_a_wrong_idea() -> None:
    def engaged(p: dict[str, Any]) -> None:
        p["concepts"][1].update(outcome="not_observed", evidence_turn=None)

    assert _verify(engaged) == [
        "concepts[c2]: a concept with an initial misconception was observed; "
        "do not use 'not_observed'"
    ]


def test_evidence_turn_is_required_unless_not_observed() -> None:
    def missing(p: dict[str, Any]) -> None:
        p["concepts"][0]["evidence_turn"] = None

    def not_observed(p: dict[str, Any]) -> None:
        p["concepts"][0].update(evidence_turn=None, outcome="not_observed")

    assert _verify(missing) == [
        "concepts[c1].evidence_turn: required unless the outcome is 'not_observed'"
    ]
    assert _verify(not_observed) == []


def test_a_known_misconception_alias_resolves_on_the_right_concept() -> None:
    def gesek_wrong_idea(p: dict[str, Any]) -> None:
        p["concepts"][0].update(outcome="misconception", misconception="m2")

    payload = scoring_payload()
    gesek_wrong_idea(payload)
    evaluation, errors = verify(draft(payload), make_session(), LIMITS)
    assert errors == [] and evaluation is not None
    assert evaluation.concept_results[0].concept_id == T_GESEK
    assert evaluation.concept_results[0].misconception_id == M_KASAR


def test_turn_quality_covers_every_answered_turn_once_with_zero_to_four() -> None:
    def gaps(p: dict[str, Any]) -> None:
        p["turn_quality"] = p["turn_quality"][:4]
        p["turn_quality"][0]["quality"] = 7

    assert _verify(gaps) == [
        "turn_quality: rate exactly these turns once each: t0, t1, t2, t3, t4",
        "turn_quality[t0]: must be 0 to 4",
    ]


def test_the_summary_has_no_digits() -> None:
    def counted(p: dict[str, Any]) -> None:
        p["summary"] = "Siswa berubah pikiran 2 kali."

    assert _verify(counted) == ["summary: write no digits; counts are never written by the model"]


def test_a_student_who_only_answered_the_opening_question_can_be_scored() -> None:
    session = make_session(turns=make_turns([ANSWERS[0]]))
    payload = scoring_payload()
    for score in payload["scores"]:
        score["evidence"] = [{"turn": "t0", "quote": "gayanya habis"}]
    payload["concepts"][0].update(evidence_turn=None, outcome="not_observed")
    payload["concepts"][1].update(evidence_turn="t0", outcome="misconception", misconception="m1")
    payload["turn_quality"] = [{"turn": "t0", "quality": 1}]
    evaluation, errors = verify(draft(payload), session, LIMITS)
    assert errors == [] and evaluation is not None
    assert {e.turn_id for s in evaluation.scores for e in s.evidence} == {TURN_IDS[0]}
