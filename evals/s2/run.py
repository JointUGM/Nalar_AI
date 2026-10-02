"""Generate an S2 teacher-review artifact with recorded cost and provenance.

    uv run python -m evals.s2.run
    uv run python -m evals.s2.run --input mission.json --ungrounded

The default input is authored synthetic material. A passing smoke run is not the
20-mission teacher quality gate or the 10-objective paired grounding evaluation.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from nalar_ai.container import build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s2_mission_designer.application.design_mission import (
    DesignMissionUseCase,
    SelectTargetsUseCase,
)
from nalar_ai.subsystems.s2_mission_designer.application.schemas import (
    GenerateMissionIn,
    SelectTargetsIn,
)

ROOT = Path(__file__).parent


async def run(input_path: Path, output: Path, *, ungrounded: bool) -> bool:
    body = GenerateMissionIn.model_validate_json(
        await asyncio.to_thread(input_path.read_text, encoding="utf-8")
    )
    if ungrounded:
        body = body.model_copy(update={"paragraphs": []})
    container = build_container(get_settings())
    ledger = UsageLedger(f"s2-smoke-{uuid4()}", container.settings.max_cost_usd_per_request)
    start = time.perf_counter()
    report: dict[str, Any] = {"input": str(input_path), "ungrounded": ungrounded, "success": False}
    try:
        selected = await SelectTargetsUseCase(container.llm, container.s2).execute(
            SelectTargetsIn(learning_objective=body.learning_objective, concepts=body.targets),
            ledger,
        )
        report["selection_s"] = round(time.perf_counter() - start, 2)
        ids = set(selected.target_concept_ids)
        body = body.model_copy(
            update={
                "targets": [t for t in body.targets if t.id in ids],
                "misconceptions": [m for m in body.misconceptions if m.concept_id in ids],
            }
        )
        result = await DesignMissionUseCase(container.llm, container.s2, container.tokens).execute(
            body, ledger
        )
        report.update(success=True, result=result.model_dump(mode="json"))
    except Exception as error:
        report.update(error_type=type(error).__name__, message=str(error))
    finally:
        report.update(
            elapsed_s=round(time.perf_counter() - start, 2),
            recorded_cost_usd=round(ledger.total_cost_usd, 6),
            invocations=[asdict(i) for i in ledger.records],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            output.write_text,
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        await container.aclose()
    print(json.dumps({k: v for k, v in report.items() if k not in {"result", "invocations"}}))
    print(f"Artifact: {output}")
    return bool(report["success"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "datasets" / "synthetic_motion.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / f"{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    )
    parser.add_argument("--ungrounded", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    raise SystemExit(
        0 if asyncio.run(run(args.input, args.output, ungrounded=args.ungrounded)) else 1
    )


if __name__ == "__main__":
    main()
