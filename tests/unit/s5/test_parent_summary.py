from dataclasses import replace

import pytest

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from nalar_ai.shared.enums import AiPurpose, ModelTier
from nalar_ai.shared.errors import UpstreamUnavailableError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s5_insight_synthesizer.application.parent_summary import (
    ParentSummaryResult,
    ParentSummaryUseCase,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import ParentSummaryInput
from tests.support.s1 import MODELS
from tests.support.s5 import (
    CONFIG,
    SUMMARY,
    make_gateway,
    make_parent_input,
    make_policy,
    summary_reply,
)

SCORED = summary_reply(SUMMARY + " Nilai Ananda di atas rata-rata teman sekelas.")


async def _run(
    llm: ScriptedLLM, data: ParentSummaryInput | None = None, ledger: UsageLedger | None = None
) -> ParentSummaryResult:
    use_case = ParentSummaryUseCase(llm=make_gateway(llm), config=CONFIG, policy=make_policy())
    return await use_case.execute(data or make_parent_input(), ledger or UsageLedger("r", 1.0))


async def test_a_clean_summary_is_used_as_written() -> None:
    llm = ScriptedLLM([summary_reply()])
    ledger = UsageLedger("r", 1.0)
    result = await _run(llm, ledger=ledger)
    assert (result.content, result.source, result.retried) == (SUMMARY, "model", False)
    (record,) = ledger.records
    assert record.purpose is AiPurpose.PARENT_SUMMARY
    assert record.model == MODELS[ModelTier.FAST]


async def test_the_prompt_fences_teacher_text_and_describes_each_outcome() -> None:
    llm = ScriptedLLM([summary_reply()])
    await _run(llm)
    task = llm.requests[0].blocks[-1].text
    assert "<mission>\nGaya dan Gerak\n</mission>" in task
    assert "- <concept>\nGaya gesek\n</concept>\n  understood" in task
    assert (
        "still holds a mistaken idea; the idea: "
        "<idea>\nBenda berat selalu jatuh lebih cepat\n</idea>"
    ) in task
    assert "<teacher_summary>\nSiswa menjelaskan gesekan" in task


async def test_a_blocked_summary_gets_one_retry_with_the_problems() -> None:
    llm = ScriptedLLM([SCORED, summary_reply()])
    result = await _run(llm)
    assert result.source == "model" and result.retried is True
    retry = llm.requests[1].blocks[-1].text
    assert "- do not mention scores, cheating or copying (found 'nilai')" in retry


async def test_blocked_twice_falls_back_to_the_template() -> None:
    llm = ScriptedLLM([SCORED, "not json"])
    result = await _run(llm)
    assert result.source == "template" and result.retried is True
    assert result.content == CONFIG.template.render(make_parent_input().concepts)


async def test_a_provider_outage_never_freezes_a_template() -> None:
    llm = ScriptedLLM([TransientLLMError("503"), TransientLLMError("503")])
    with pytest.raises(UpstreamUnavailableError):
        await _run(llm)


async def test_a_concept_name_with_a_digit_does_not_force_the_template() -> None:
    data = make_parent_input()
    renamed = replace(data.concepts[0], name="Hukum Newton 1")
    data = replace(data, concepts=(renamed, data.concepts[1]))
    llm = ScriptedLLM([summary_reply(SUMMARY + " Ananda paham Hukum Newton 1.")])
    assert (await _run(llm, data)).source == "model"


async def test_a_missing_teacher_summary_is_said_plainly() -> None:
    llm = ScriptedLLM([summary_reply()])
    await _run(llm, make_parent_input(evaluation_summary=None))
    task = llm.requests[0].blocks[-1].text
    assert "The teacher's summary of the session, if any:\n(none)" in task
