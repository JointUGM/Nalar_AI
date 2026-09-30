import pytest

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from nalar_ai.shared.enums import AiPurpose, ModelTier
from nalar_ai.shared.errors import (
    InvalidInputError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s5_insight_synthesizer.application.class_insight import (
    ClassInsightUseCase,
    InsightResult,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import ClassCounts
from tests.support.s1 import MODELS
from tests.support.s5 import (
    CONFIG,
    M_HABIS,
    insight_reply,
    make_counts,
    make_gateway,
    make_policy,
)

LEAKY = insight_reply(narrative="Sebagian besar siswa, yaitu 12 anak, masih bingung.")


async def _run(
    llm: ScriptedLLM, counts: ClassCounts | None = None, ledger: UsageLedger | None = None
) -> InsightResult:
    use_case = ClassInsightUseCase(llm=make_gateway(llm), config=CONFIG, policy=make_policy())
    return await use_case.execute(counts or make_counts(), ledger or UsageLedger("r", 1.0))


async def test_a_clean_insight_comes_back_with_ids() -> None:
    llm = ScriptedLLM([insight_reply()])
    ledger = UsageLedger("r", 1.0)
    result = await _run(llm, ledger=ledger)
    assert result.retried is False
    assert result.insight.narrative.startswith(f"{{{{count:{M_HABIS}}}}} dari {{{{total}}}}")
    (record,) = ledger.records
    assert record.purpose is AiPurpose.CLASS_MAP_INSIGHT
    assert record.model == MODELS[ModelTier.QUALITY]


async def test_the_prompt_carries_fenced_teacher_text_and_the_counts() -> None:
    llm = ScriptedLLM([insight_reply()])
    await _run(llm)
    task = llm.requests[0].blocks[-1].text
    assert "<mission>\nGaya dan Gerak\n</mission>" in task
    assert "Students with a finished, scored session: 28" in task
    assert (
        "c1: <concept>\nGaya gesek\n</concept>\n  understood 9, developing 7, not observed 2"
    ) in task
    assert (
        "  m1: <idea>\nGaya bisa habis seperti bensin\n</idea>\n"
        "    holds it now 10, changed their mind during the session 4"
    ) in task
    assert "Problems with your previous attempt, if any:\n(none)" in task


async def test_a_leaky_draft_gets_one_retry_with_the_problems() -> None:
    llm = ScriptedLLM([LEAKY, insight_reply()])
    result = await _run(llm)
    assert result.retried is True
    assert "- narrative: write no digits" in llm.requests[1].blocks[-1].text


async def test_two_leaky_drafts_fail_with_ai_output_invalid() -> None:
    llm = ScriptedLLM([LEAKY, LEAKY])
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(OutputValidationError) as error:
        await _run(llm, ledger=ledger)
    assert "narrative: write no digits" in error.value.details["errors"]
    assert len(ledger.records) == 2


async def test_invalid_json_counts_as_a_failed_try() -> None:
    llm = ScriptedLLM(["not json", insight_reply()])
    assert (await _run(llm)).retried is True


async def test_a_provider_outage_propagates() -> None:
    llm = ScriptedLLM([TransientLLMError("503"), TransientLLMError("503")])
    with pytest.raises(UpstreamUnavailableError):
        await _run(llm)


async def test_inconsistent_counts_fail_before_any_model_call() -> None:
    llm = ScriptedLLM([insight_reply()])
    with pytest.raises(InvalidInputError):
        await _run(llm, counts=make_counts(total=5))
    assert llm.requests == []
