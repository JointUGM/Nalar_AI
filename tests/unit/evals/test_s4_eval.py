import json

from evals.s4.data import SAMPLE_SESSIONS, load_cases
from evals.s4.metrics import consistency, teacher_agreement
from evals.s4.run import run_evaluations
from nalar_ai.container import build_container
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.settings import Settings
from nalar_ai.shared.enums import RubricDimension as D
from nalar_ai.subsystems.s4_session_evaluator.domain.session import validate_session
from tests.support.s4 import by_prompt, reflection_reply, scoring_payload, scoring_reply

ALL_TWO = dict.fromkeys(D, 2)


def test_consistency_counts_dimensions_identical_in_every_run() -> None:
    changed = {**ALL_TWO, D.TRANSFER: 3}
    assert consistency([[ALL_TWO, ALL_TWO, ALL_TWO]]) == 1.0
    assert consistency([[ALL_TWO, ALL_TWO, changed]]) == 0.75
    assert consistency([[ALL_TWO]]) is None  # one run says nothing about consistency


def test_teacher_agreement_reports_exact_and_within_one() -> None:
    teacher = {D.CLAIM: 2, D.EVIDENCE: 2, D.MECHANISM: 0, D.TRANSFER: 4}
    assert teacher_agreement([(ALL_TWO, teacher)]) == {"exact": 0.5, "within_one": 0.5}
    assert teacher_agreement([]) is None


def test_the_sample_sessions_load_and_are_valid() -> None:
    (case,) = load_cases(SAMPLE_SESSIONS)
    assert validate_session(case.session) == []
    assert case.teacher_levels == {D.CLAIM: 3, D.EVIDENCE: 2, D.MECHANISM: 2, D.TRANSFER: 3}


async def test_a_consistent_scorer_that_matches_the_teacher_passes(settings: Settings) -> None:
    llm = by_prompt(score=[scoring_reply()] * 3, reflect=[reflection_reply()] * 3)
    container = build_container(settings, llm_port=llm, embedding_port=HashingEmbedder())
    report = await run_evaluations(container, load_cases(SAMPLE_SESSIONS), runs=3)
    assert report["consistency"] == 1.0
    assert report["teacher_agreement"] == {"exact": 1.0, "within_one": 1.0}
    assert report["reflection_fallback_rate"] == 0.0
    assert report["failures"] == []
    assert report["passed"] is True


async def test_an_inconsistent_scorer_fails_the_gate(settings: Settings) -> None:
    lower = scoring_payload()
    for score in lower["scores"]:
        score["level"] = 0
    llm = by_prompt(score=[scoring_reply(), json.dumps(lower)], reflect=[reflection_reply()] * 2)
    container = build_container(settings, llm_port=llm, embedding_port=HashingEmbedder())
    report = await run_evaluations(container, load_cases(SAMPLE_SESSIONS), runs=2)
    assert report["consistency"] == 0.0
    assert report["gates"]["consistency"] is False
    assert report["passed"] is False
