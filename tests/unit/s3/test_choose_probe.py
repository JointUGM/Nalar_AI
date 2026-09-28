import json
from typing import Any

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from nalar_ai.shared.enums import AiPurpose, MoveReasonCode, PlannerMode
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.choose_probe import (
    ChoiceFailure,
    ChooseProbeUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import plan_move
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView
from nalar_ai.subsystems.s3_socratic_prober.domain.validation import ModelChoice
from tests.support.s3 import M_HABIS, T_LEMBAM, anchor, make_gateway, make_pack, question_id

PACK = make_pack()
VIEW = SessionView(
    (anchor("kelerengnya berhenti karena gayanya habis", misconceptions=(M_HABIS,)),)
)
CLASSIFICATION = Classification(
    AnswerType.MISCONCEPTION,
    ClassificationSource.MODEL,
    (M_HABIS,),
    frustration=True,
    key_phrase="gayanya habis",
)
PLAN = plan_move(CLASSIFICATION, VIEW, PACK, mode=PlannerMode.HYBRID)
# The sample bank is aliased in pack order: q11 is lembam/counter_example.
COUNTER_ALIAS = "q11"


def _reply(**changes: Any) -> str:
    reply = {
        "move": "counter_example",
        "question_id": COUNTER_ALIAS,
        "reason_code": "default",
        "reason": "",
        "question": "Kamu bilang gayanya habis. Kenapa pesawat luar angkasa tetap melaju?",
    }
    return json.dumps({**reply, **changes})


def _use_case(llm: ScriptedLLM) -> ChooseProbeUseCase:
    return ChooseProbeUseCase(llm=make_gateway(llm), transcript_turns=6, max_answer_chars=1500)


async def _choose(llm: ScriptedLLM, timeout_s: float | None = 2.5) -> ModelChoice | ChoiceFailure:
    return await _use_case(llm).execute(
        PACK.writer_view(), VIEW, CLASSIFICATION, PLAN, UsageLedger("r", 1.0), timeout_s=timeout_s
    )


async def test_the_choice_is_resolved_to_bank_ids() -> None:
    llm = ScriptedLLM([_reply()])
    ledger = UsageLedger("r", 1.0)
    choice = await _use_case(llm).execute(
        PACK.writer_view(), VIEW, CLASSIFICATION, PLAN, ledger, timeout_s=2.5
    )
    assert choice == ModelChoice(
        move=M.COUNTER_EXAMPLE,
        question_id=question_id(T_LEMBAM, M.COUNTER_EXAMPLE),
        reason_code=MoveReasonCode.DEFAULT,
        reason="",
        question="Kamu bilang gayanya habis. Kenapa pesawat luar angkasa tetap melaju?",
    )
    (record,) = ledger.records
    assert record.purpose is AiPurpose.PROBE_PLAN
    assert record.model == "claude-sonnet-5"
    assert llm.requests[0].effort == "low"


async def test_the_writer_never_sees_the_answer() -> None:
    llm = ScriptedLLM([_reply()])
    await _choose(llm)
    (request,) = llm.requests
    everything = request.system + "\n".join(block.text for block in request.blocks)
    assert PACK.reference_reasoning not in everything
    for target in PACK.targets:
        assert target.name not in everything
        assert target.description not in everything
    for term in ("gesekan", "kelembaman", "inersia", "hukum newton"):
        assert term not in everything.casefold()


async def test_the_task_lists_allowed_moves_default_first_and_signals() -> None:
    llm = ScriptedLLM([_reply()])
    await _choose(llm)
    context, task = llm.requests[0].blocks[0].text, llm.requests[0].blocks[1].text
    assert llm.requests[0].blocks[0].cache
    assert f"[{COUNTER_ALIAS}] (for m1) Kalau dorongannya habis" in context
    assert "- counter_example (default): q11" in task
    assert "- explain_mechanism: q15" in task
    assert "Target concept this turn: c2" in task
    assert "Wrong ideas in the latest answer: m1" in task
    assert "Signals: the student sounds frustrated" in task
    assert "<key_phrase>\ngayanya habis\n</key_phrase>" in task


async def test_the_cached_context_is_the_same_for_every_student() -> None:
    llm = ScriptedLLM([_reply(), _reply()])
    await _choose(llm)
    other_view = SessionView((anchor("gatau deh, pokoknya habis"),))
    await _use_case(llm).execute(
        PACK.writer_view(),
        other_view,
        CLASSIFICATION,
        PLAN,
        UsageLedger("r", 1.0),
        timeout_s=2.5,
    )
    first, second = llm.requests
    assert first.system == second.system
    assert first.blocks[0].text == second.blocks[0].text


async def test_unparseable_output_is_invalid_and_never_repaired() -> None:
    llm = ScriptedLLM([_reply(move="give_the_answer"), _reply()])
    assert await _choose(llm) is ChoiceFailure.INVALID
    assert len(llm.requests) == 1


async def test_provider_trouble_or_no_time_is_an_error() -> None:
    assert await _choose(ScriptedLLM([TransientLLMError("529")])) is ChoiceFailure.ERROR
    llm = ScriptedLLM()
    assert await _choose(llm, timeout_s=None) is ChoiceFailure.ERROR
    assert llm.requests == []


async def test_an_unknown_question_alias_resolves_to_none() -> None:
    choice = await _choose(ScriptedLLM([_reply(question_id="q99")]))
    assert isinstance(choice, ModelChoice) and choice.question_id is None
