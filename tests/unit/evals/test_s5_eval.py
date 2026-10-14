from evals.s5.run import run_gate
from nalar_ai.container import Container
from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.s5 import insight_reply


async def test_the_gate_counts_valid_insights(container: Container, llm: ScriptedLLM) -> None:
    llm.queue(insight_reply())
    report = await run_gate(container, runs=1)
    assert report["insight_valid_rate"] == 1.0
    assert len(llm.requests) == 1
    assert report["passed"] is True


async def test_the_gate_fails_when_suggestions_invent_counts(
    container: Container, llm: ScriptedLLM
) -> None:
    llm.queue(
        *[insight_reply(suggestions=["Ajak 12 siswa menjelaskan."])] * 2,
    )
    report = await run_gate(container, runs=1)
    assert report["insight_valid_rate"] == 0.0
    assert report["passed"] is False


async def test_the_gate_covers_classes_with_no_active_ideas_and_a_nonfirst_active_idea(
    container: Container, llm: ScriptedLLM
) -> None:
    for _ in range(3):
        llm.queue(insight_reply(clusters=[], suggestions=[]))
    report = await run_gate(container, runs=3)
    tasks = [
        r.blocks[-1].text
        for r in llm.requests
        if r.json_schema and "clusters" in r.json_schema["properties"]
    ]
    assert report["passed"] and report["class_cases"] == 3
    assert "Allowed idea aliases for clusters: (none)" in tasks[1]
    assert "Allowed idea aliases for clusters: m2" in tasks[2]
