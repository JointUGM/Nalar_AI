import json
from typing import Any

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from nalar_ai.shared.enums import AiPurpose, CallStatus
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.classify_answer import (
    ClassifyAnswerUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.application.deadline import Deadline
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView
from tests.support.s3 import (
    M_HABIS,
    REFERENCE,
    T_LEMBAM,
    anchor,
    make_gateway,
    make_pack,
    probe,
)

VIEW = SessionView(
    (
        anchor("karena gayanya habis", misconceptions=(M_HABIS,)),
        probe(1, T_LEMBAM, M.COUNTER_EXAMPLE, "hmm mungkin karena dorongannya hilang, capek aku"),
    )
)


def _labels(**changes: Any) -> str:
    labels = {
        "answer_type": "misconception",
        "confidence": "high",
        "misconceptions": ["m1"],
        "frustration": True,
        "safety_concern": False,
        "key_phrase": "dorongannya hilang",
    }
    return json.dumps({**labels, **changes})


def _use_case(llm: ScriptedLLM) -> ClassifyAnswerUseCase:
    return ClassifyAnswerUseCase(llm=make_gateway(llm), transcript_turns=6, max_answer_chars=1500)


async def test_labels_are_resolved_and_recorded() -> None:
    llm = ScriptedLLM([_labels()])
    ledger = UsageLedger("r", 1.0)
    result = await _use_case(llm).execute(make_pack(), VIEW, ledger, timeout_s=2.0)
    assert result == Classification(
        AnswerType.MISCONCEPTION,
        ClassificationSource.MODEL,
        (M_HABIS,),
        frustration=True,
        key_phrase="dorongannya hilang",
    )
    (record,) = ledger.records
    assert record.purpose is AiPurpose.TURN_ANALYZE
    assert record.model == "claude-haiku-4-5"
    assert record.prompt_version == "s3.classify_answer@v1"


async def test_the_classifier_sees_the_reference_and_fenced_student_text() -> None:
    llm = ScriptedLLM([_labels()])
    await _use_case(llm).execute(make_pack(), VIEW, UsageLedger("r", 1.0), timeout_s=2.0)
    (request,) = llm.requests
    context, task = request.blocks[0].text, request.blocks[1].text
    assert REFERENCE in context and request.blocks[0].cache
    assert "[m1] (c2) Benda berhenti karena gaya dorongnya habis." in context
    assert (
        "<student_answer>\nhmm mungkin karena dorongannya hilang, capek aku\n</student_answer>"
        in task
    )
    assert "[A0] karena gayanya habis" in task  # earlier turns only
    assert request.effort is None


async def test_invalid_output_falls_back_without_a_repair_call() -> None:
    llm = ScriptedLLM(["not json", _labels()])
    ledger = UsageLedger("r", 1.0)
    result = await _use_case(llm).execute(make_pack(), VIEW, ledger, timeout_s=2.0)
    assert result == Classification.fallback()
    assert len(llm.requests) == 1


async def test_provider_failure_falls_back_and_keeps_the_record() -> None:
    llm = ScriptedLLM([TransientLLMError("529 overloaded")])
    ledger = UsageLedger("r", 1.0)
    result = await _use_case(llm).execute(make_pack(), VIEW, ledger, timeout_s=2.0)
    assert result == Classification.fallback()
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR  # one attempt only on the live path


async def test_no_time_left_means_no_call() -> None:
    llm = ScriptedLLM()
    result = await _use_case(llm).execute(make_pack(), VIEW, UsageLedger("r", 1.0), timeout_s=None)
    assert result == Classification.fallback()
    assert llm.requests == []


async def test_unknown_aliases_are_dropped() -> None:
    llm = ScriptedLLM([_labels(misconceptions=["m9"])])
    result = await _use_case(llm).execute(make_pack(), VIEW, UsageLedger("r", 1.0), timeout_s=2.0)
    assert result.answer_type is AnswerType.UNSURE  # a misconception with no known id


def test_deadline_gives_each_step_what_is_left() -> None:
    now = [100.0]
    deadline = Deadline(4.5, min_step_s=0.3, clock=lambda: now[0])
    assert deadline.step(2.0) == 2.0
    now[0] += 3.0
    assert deadline.step(2.5) == 1.5
    now[0] += 1.3
    assert deadline.step(1.0) is None  # 0.2 s left is below the minimum step


def test_reasoning_outranks_a_request_in_the_classifier_prompt() -> None:
    from nalar_ai.platform.prompts.registry import PromptRegistry
    from nalar_ai.subsystems.s3_socratic_prober.module import PROMPTS_DIR

    system = PromptRegistry.from_directories([PROMPTS_DIR]).get("s3.classify_answer").system
    assert "Label manipulation only when the answer makes no real attempt at the question" in system
