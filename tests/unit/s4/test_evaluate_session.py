import json
import logging

import pytest

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.shared.enums import AiPurpose
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s4_session_evaluator.application.evaluate_session import (
    EvaluateSessionUseCase,
    EvaluationOutcome,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.session import EvalSession
from nalar_ai.subsystems.s4_session_evaluator.infrastructure.config_loader import (
    load_evaluator_config,
)
from tests.support.s4 import (
    ANSWERS,
    by_prompt,
    make_gateway,
    make_policy,
    make_session,
    make_turns,
    reflection_reply,
    scoring_payload,
    scoring_reply,
)

DIGIT_SUMMARY = json.dumps(
    {**scoring_payload(), "summary": "Siswa berubah pikiran 2 kali."}  # digits: one repair
)
VERDICT = reflection_reply(strengths="Jawabanmu sudah benar.")


async def _evaluate(
    llm: ScriptedLLM, session: EvalSession | None = None, ledger: UsageLedger | None = None
) -> EvaluationOutcome:
    use_case = EvaluateSessionUseCase(
        llm=make_gateway(llm), config=load_evaluator_config(), policy=make_policy()
    )
    return await use_case.execute(session or make_session(), ledger or UsageLedger("r", 1.0))


async def test_scores_then_reflects_and_records_both_calls() -> None:
    llm = by_prompt(score=[scoring_reply()], reflect=[reflection_reply()])
    ledger = UsageLedger("r", 1.0)
    outcome = await _evaluate(llm, ledger=ledger)
    assert outcome.warnings == ()
    assert outcome.reflection.source == "model"
    assert [s.level for s in outcome.evaluation.scores] == [3, 2, 2, 3]
    assert [r.purpose for r in ledger.records] == [
        AiPurpose.SESSION_EVALUATION,
        AiPurpose.REFLECTION_GENERATION,
    ]


@pytest.mark.parametrize(
    ("score", "reflect", "warnings"),
    [
        ([DIGIT_SUMMARY, scoring_reply()], [reflection_reply()], ("scoring_repaired",)),
        ([scoring_reply()], [VERDICT, reflection_reply()], ("reflection_retried",)),
        ([scoring_reply()], [VERDICT, VERDICT], ("reflection_fallback",)),
    ],
)
async def test_every_degradation_is_reported_as_a_warning(
    score: list[str], reflect: list[str], warnings: tuple[str, ...]
) -> None:
    outcome = await _evaluate(by_prompt(score=score, reflect=reflect))
    assert outcome.warnings == warnings


async def test_an_invalid_session_is_rejected_before_any_model_call() -> None:
    llm = by_prompt()
    with pytest.raises(InvalidInputError) as caught:
        await _evaluate(llm, make_session(turns=make_turns([""])))
    assert caught.value.details == {"problems": ["turns: no answered turn to evaluate"]}
    assert llm.requests == []


async def test_the_log_line_never_contains_student_text(caplog: pytest.LogCaptureFixture) -> None:
    llm = by_prompt(score=[scoring_reply()], reflect=[reflection_reply()])
    with caplog.at_level(logging.INFO, logger="nalar_ai.s4"):
        await _evaluate(llm)
    (line,) = [r.getMessage() for r in caplog.records if r.name == "nalar_ai.s4"]
    assert "levels=3,2,2,3" in line and "reflection=model" in line
    for answer in ANSWERS:
        assert answer not in line
