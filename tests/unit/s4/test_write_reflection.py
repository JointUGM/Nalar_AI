import pytest

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from nalar_ai.shared.enums import AiPurpose
from nalar_ai.shared.errors import UpstreamUnavailableError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s4_session_evaluator.application.write_reflection import (
    ReflectionResult,
    WriteReflectionUseCase,
)
from nalar_ai.subsystems.s4_session_evaluator.infrastructure.config_loader import (
    load_evaluator_config,
)
from tests.support.s3 import ANCHOR, REFERENCE
from tests.support.s4 import (
    make_evaluation,
    make_gateway,
    make_pack,
    make_policy,
    make_session,
    reflection_reply,
)

CONFIG = load_evaluator_config()
VERDICT = reflection_reply(strengths="Jawabanmu tentang lantai sudah benar.")


async def _run(llm: ScriptedLLM, ledger: UsageLedger | None = None) -> ReflectionResult:
    use_case = WriteReflectionUseCase(llm=make_gateway(llm), config=CONFIG, policy=make_policy())
    return await use_case.execute(
        make_session(), make_evaluation(), ledger or UsageLedger("r", 1.0)
    )


async def test_a_clean_reflection_is_used_as_written() -> None:
    llm = ScriptedLLM([reflection_reply()])
    ledger = UsageLedger("r", 1.0)
    result = await _run(llm, ledger)
    assert result.source == "model" and result.retried is False
    assert result.parts.content.startswith("Kamu memakai contoh lantai dan es")
    (record,) = ledger.records
    assert record.purpose is AiPurpose.REFLECTION_GENERATION
    assert record.model == "claude-haiku-4-5"


async def test_the_writer_never_sees_the_answer_or_the_concepts() -> None:
    llm = ScriptedLLM([VERDICT, VERDICT])
    await _run(llm)
    pack = make_pack()
    for request in llm.requests:
        text = request.system + "".join(block.text for block in request.blocks)
        assert REFERENCE not in text
        for target in pack.targets:
            assert target.name not in text and target.description not in text
        for misconception in pack.misconceptions:
            assert misconception.statement not in text
        for term in ("kelembaman", "inersia", "hukum newton", "gaya gesek"):
            assert term not in text.lower()  # the student never wrote these


async def test_the_writer_gets_the_changed_turn_and_the_strongest_quotes() -> None:
    llm = ScriptedLLM([reflection_reply()])
    await _run(llm)
    task = llm.requests[0].blocks[-1].text
    assert "Where the student rethought an idea (turn ids): t1" in task
    assert "- t2: <student_quote>\nada gesekan jadi pelan-pelan berhenti\n</student_quote>" in task
    assert "- t3: <student_quote>\nDi es lebih jauh karena licin\n</student_quote>" in task
    assert "Problems with your previous attempt, if any:\n(none)" in task


async def test_a_blocked_reflection_gets_one_fresh_retry_with_the_problems() -> None:
    llm = ScriptedLLM([VERDICT, reflection_reply()])
    result = await _run(llm)
    assert result.source == "model" and result.retried is True
    retry = llm.requests[1].blocks[-1].text
    assert "- do not judge right or wrong (found 'benar')" in retry


async def test_blocked_twice_falls_back_to_the_template() -> None:
    llm = ScriptedLLM([VERDICT, VERDICT])
    result = await _run(llm)
    assert result.source == "template" and result.retried is True
    assert result.parts == CONFIG.template
    assert len(llm.requests) == 2


async def test_invalid_json_counts_as_a_blocked_attempt() -> None:
    llm = ScriptedLLM(["not json", reflection_reply()])
    result = await _run(llm)
    assert result.source == "model" and result.retried is True


async def test_a_provider_outage_fails_the_evaluation_instead_of_freezing_a_template() -> None:
    llm = ScriptedLLM([TransientLLMError("503"), TransientLLMError("503")])
    with pytest.raises(UpstreamUnavailableError):
        await _run(llm)


async def test_the_reflection_prompt_fences_the_opening_and_the_questions() -> None:
    llm = ScriptedLLM([reflection_reply()])
    await _run(llm)
    task = llm.requests[0].blocks[-1].text
    assert f"<anchor>\n{ANCHOR}\n</anchor>" in task
    assert task.count("<question>\n") == len(make_session().turns)
