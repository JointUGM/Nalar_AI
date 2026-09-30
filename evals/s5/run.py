"""S5 eval gate (AI-12; design doc §11). Real runs cost money and need a configured .env.

    uv run python -m evals.s5.run --runs 3

Per run: one class insight and one parent summary per sample student (about $0.03).
Gates: insight valid within two tries >= 95%; summary written by the model >= 95%.
Exit code 1 when a gate fails.
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any
from uuid import UUID

from evals.s3.metrics import percentile
from nalar_ai.container import Container, build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.enums import ConceptOutcome as O
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s5_insight_synthesizer.application.class_insight import (
    ClassInsightUseCase,
)
from nalar_ai.subsystems.s5_insight_synthesizer.application.parent_summary import (
    ParentSummaryUseCase,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import (
    ClassCounts,
    ConceptCount,
    MisconceptionCount,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    ParentConcept,
    ParentSummaryInput,
)

GATE = 0.95
TITLE = "Gaya dan Gerak"
GESEK, LEMBAM = "Gaya gesek", "Kelembaman"
HABIS = "Gaya bisa habis seperti bensin"
BERAT = "Benda berat selalu jatuh lebih cepat"

SAMPLE_CLASS = ClassCounts(
    mission_title=TITLE,
    total=28,
    incomplete=3,
    concepts=(
        ConceptCount(
            UUID(int=1), GESEK, 9, 7, 2, (MisconceptionCount(UUID(int=11), HABIS, 10, 4),)
        ),
        ConceptCount(
            UUID(int=2), LEMBAM, 12, 8, 2, (MisconceptionCount(UUID(int=12), BERAT, 6, 1),)
        ),
    ),
)


def _student(gesek: O, lembam: O, *, resolved: bool = False) -> ParentSummaryInput:
    held = gesek is O.MISCONCEPTION or resolved
    return ParentSummaryInput(
        mission_title=TITLE,
        concepts=(
            ParentConcept(GESEK, gesek, HABIS if held else None, resolved),
            ParentConcept(LEMBAM, lembam, BERAT if lembam is O.MISCONCEPTION else None, False),
        ),
        evaluation_summary=None,
    )


SAMPLE_STUDENTS = (
    _student(O.MASTERED, O.MASTERED),
    _student(O.MASTERED, O.MISCONCEPTION),
    _student(O.MISCONCEPTION, O.DEVELOPING),
    _student(O.DEVELOPING, O.NOT_OBSERVED, resolved=True),
    _student(O.NOT_OBSERVED, O.NOT_OBSERVED),
)


async def run_gate(container: Container, *, runs: int) -> dict[str, Any]:
    insight = ClassInsightUseCase(
        llm=container.llm, config=container.s5.config, policy=container.s5.policy
    )
    summary = ParentSummaryUseCase(
        llm=container.llm, config=container.s5.config, policy=container.s5.policy
    )
    cap = container.settings.max_cost_usd_per_request
    valid = model = 0
    insight_s: list[float] = []
    summary_s: list[float] = []
    cost = 0.0
    for run in range(runs):
        ledger = UsageLedger(f"s5-insight-{run}", cap)
        start = time.perf_counter()
        try:
            await insight.execute(SAMPLE_CLASS, ledger)
            valid += 1
        except OutputValidationError:
            pass
        insight_s.append(time.perf_counter() - start)
        cost += ledger.total_cost_usd
        for index, student in enumerate(SAMPLE_STUDENTS):
            ledger = UsageLedger(f"s5-summary-{run}-{index}", cap)
            start = time.perf_counter()
            result = await summary.execute(student, ledger)
            summary_s.append(time.perf_counter() - start)
            model += result.source == "model"
            cost += ledger.total_cost_usd
    insight_rate = valid / runs
    summary_rate = model / (runs * len(SAMPLE_STUDENTS))
    return {
        "runs": runs,
        "insight_valid_rate": insight_rate,
        "summary_model_rate": summary_rate,
        "insight_p95_s": round(percentile(insight_s, 95), 2),
        "summary_p95_s": round(percentile(summary_s, 95), 2),
        "cost_usd": round(cost, 4),
        "passed": insight_rate >= GATE and summary_rate >= GATE,
    }


async def _main(runs: int) -> int:
    container = build_container(get_settings())
    try:
        report = await run_gate(container, runs=runs)
    finally:
        await container.aclose()
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    sys.exit(asyncio.run(_main(parser.parse_args().runs)))
