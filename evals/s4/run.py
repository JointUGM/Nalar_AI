"""S4 eval gate (AI-12). Real runs cost money and need a configured .env.

    uv run python -m evals.s4.run --sessions evals/s4/datasets/<file>.json --runs 3

Every session is evaluated --runs times. Reports consistency (runs >= 2), agreement with
`teacher_levels` where given, the reflection fallback rate, cost and latency.
At least two successful runs are required for consistency. Missing teacher scores are
reported as a skipped gate. Exit code 1 when a required gate fails or lacks evidence.
"""

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.s3.metrics import percentile
from evals.s4.data import SAMPLE_RUBRIC, SAMPLE_SESSIONS, EvalCase, load_cases
from evals.s4.metrics import GATES, Levels, consistency, teacher_agreement
from nalar_ai.container import Container, build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.errors import NalarAIError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s4_session_evaluator.application.evaluate_session import (
    EvaluateSessionUseCase,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.quotes import find_quote


async def run_evaluations(
    container: Container, cases: Sequence[EvalCase], *, runs: int
) -> dict[str, Any]:
    use_case = EvaluateSessionUseCase(
        llm=container.llm, config=container.s4.config, policy=container.s4.policy
    )
    levels: list[list[Levels]] = []
    pairs: list[tuple[Levels, Levels]] = []
    failures: list[dict[str, str]] = []
    costs: list[float] = []
    latencies: list[float] = []
    reflections = fallbacks = unverified_quotes = 0
    for case in cases:
        answers = {turn.turn_id: turn.answer_text for turn in case.session.turns}
        case_levels: list[Levels] = []
        for run in range(runs):
            ledger = UsageLedger(f"eval-s4-{case.name}-{run}", 1.0)
            started = time.perf_counter()
            try:
                outcome = await use_case.execute(case.session, ledger)
            except NalarAIError as exc:
                failures.append({"session": case.name, "run": str(run), "error": exc.code})
                continue
            finally:
                costs.append(ledger.total_cost_usd)
            latencies.append(time.perf_counter() - started)
            scores = outcome.evaluation.scores
            case_levels.append({score.dimension: score.level for score in scores})
            unverified_quotes += sum(
                find_quote(answers[e.turn_id], e.quote) != e.quote
                for score in scores
                for e in score.evidence
            )
            reflections += 1
            fallbacks += outcome.reflection.source == "template"
        levels.append(case_levels)
        if case.teacher_levels and case_levels:
            pairs.append((case_levels[0], case.teacher_levels))
    agreement = teacher_agreement(pairs)
    consistent = consistency(levels)
    gates = {
        "quotes_verbatim": unverified_quotes == 0,
        "no_failures": not failures,
        "consistency": consistent is not None and consistent >= GATES["consistency"],
    }
    skipped_gates = {}
    if agreement is not None:
        gates["teacher_within_one"] = agreement["within_one"] >= GATES["teacher_within_one"]
    else:
        skipped_gates["teacher_within_one"] = "no teacher-labelled scores"
    return {
        "sessions": len(cases),
        "runs": runs,
        "consistency": round(consistent, 4) if consistent is not None else None,
        "teacher_agreement": agreement,
        "reflection_fallback_rate": round(fallbacks / reflections, 4) if reflections else None,
        "failures": failures,
        "cost_usd_per_session": round(sum(costs) / len(costs), 6) if costs else None,
        "latency_s": {
            "p50": percentile(latencies, 50) if latencies else None,
            "p95": percentile(latencies, 95) if latencies else None,
        },
        "gates": gates,
        "skipped_gates": skipped_gates,
        "passed": all(gates.values()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S4 eval gate (AI-12)")
    parser.add_argument("--sessions", type=Path, default=SAMPLE_SESSIONS)
    parser.add_argument("--rubric", type=Path, default=SAMPLE_RUBRIC, help="if the file has none")
    parser.add_argument("--runs", type=int, default=3, help="evaluations per session")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)
    report = asyncio.run(_run(args))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("PASS" if report["passed"] else "FAIL", file=sys.stderr)
    return 0 if report["passed"] else 1


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    container = build_container(get_settings())
    try:
        cases = load_cases(args.sessions, args.rubric)
        return await run_evaluations(container, cases, runs=args.runs)
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(main())
