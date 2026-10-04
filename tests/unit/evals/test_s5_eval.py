from evals.s5.run import SAMPLE_STUDENTS, run_gate
from nalar_ai.container import Container
from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.s5 import insight_reply, summary_reply


async def test_the_gate_counts_valid_insights_and_model_summaries(
    container: Container, llm: ScriptedLLM
) -> None:
    llm.queue(insight_reply(), *[summary_reply()] * len(SAMPLE_STUDENTS))
    report = await run_gate(container, runs=1)
    assert report["insight_valid_rate"] == 1.0
    assert report["summary_model_rate"] == 1.0
    assert report["passed"] is True


async def test_the_gate_fails_when_summaries_fall_back(
    container: Container, llm: ScriptedLLM
) -> None:
    llm.queue(insight_reply(), *[summary_reply("Terlalu pendek.")] * (2 * len(SAMPLE_STUDENTS)))
    report = await run_gate(container, runs=1)
    assert report["summary_model_rate"] == 0.0
    assert report["passed"] is False


async def test_the_gate_fails_when_suggestions_invent_counts(
    container: Container, llm: ScriptedLLM
) -> None:
    llm.queue(
        *[insight_reply(suggestions=["Ajak 12 siswa menjelaskan."])] * 2,
        *[summary_reply()] * len(SAMPLE_STUDENTS),
    )
    report = await run_gate(container, runs=1)
    assert report["insight_valid_rate"] == 0.0
    assert report["passed"] is False
