"""S5 eval gate (AI-12; design doc §11). Real runs cost money and need a configured .env.

    uv run python -m evals.s5.run --runs 3

Per run: one teacher class insight.
Gate: insight valid within two tries >= 95%.
Exit code 1 when a gate fails.
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import replace
from typing import Any
from uuid import UUID

from evals.s3.metrics import percentile
from nalar_ai.container import Container, build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s5_insight_synthesizer.application.class_insight import (
    ClassInsightUseCase,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import (
    ClassCounts,
    ConceptCount,
    MisconceptionCount,
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

MASTERED_CLASS = replace(
    SAMPLE_CLASS,
    total=1,
    incomplete=0,
    concepts=tuple(
        replace(
            c,
            mastered=1,
            developing=0,
            not_observed=0,
            misconceptions=tuple(replace(m, count=0, resolved_count=1) for m in c.misconceptions),
        )
        for c in SAMPLE_CLASS.concepts
    ),
)
SPARSE_CLASS = replace(
    SAMPLE_CLASS,
    total=1,
    incomplete=0,
    concepts=tuple(
        replace(
            c,
            mastered=1 if i == 0 else 0,
            developing=0,
            not_observed=0,
            misconceptions=tuple(
                replace(m, count=0 if i == 0 else 1, resolved_count=0) for m in c.misconceptions
            ),
        )
        for i, c in enumerate(SAMPLE_CLASS.concepts)
    ),
)
SAMPLE_CLASSES = (SAMPLE_CLASS, MASTERED_CLASS, SPARSE_CLASS)


async def run_gate(container: Container, *, runs: int) -> dict[str, Any]:
    insight = ClassInsightUseCase(
        llm=container.llm, config=container.s5.config, policy=container.s5.policy
    )
    cap = container.settings.max_cost_usd_per_request
    valid = 0
    insight_s: list[float] = []
    cost = 0.0
    for run in range(runs):
        ledger = UsageLedger(f"s5-insight-{run}", cap)
        start = time.perf_counter()
        try:
            await insight.execute(SAMPLE_CLASSES[run % len(SAMPLE_CLASSES)], ledger)
            valid += 1
        except OutputValidationError:
            pass
        insight_s.append(time.perf_counter() - start)
        cost += ledger.total_cost_usd
    insight_rate = valid / runs
    return {
        "runs": runs,
        "class_cases": min(runs, len(SAMPLE_CLASSES)),
        "insight_valid_rate": insight_rate,
        "insight_p95_s": round(percentile(insight_s, 95), 2),
        "cost_usd": round(cost, 4),
        "passed": insight_rate >= GATE,
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
