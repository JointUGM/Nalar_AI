import argparse
import asyncio
import json
import re
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from evals.s2.revision_budget import BudgetedLLMPort, CampaignBudget
from nalar_ai.container import build_container
from nalar_ai.platform.llm.openai_chat_adapter import OpenAIChatAdapter
from nalar_ai.settings import get_settings
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s2_mission_designer.application.revise_mission import ReviseMissionUseCase
from nalar_ai.subsystems.s2_mission_designer.application.schemas import ReviseMissionIn

ROOT = Path(__file__).parent


async def run(args: argparse.Namespace) -> bool:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.campaign):
        raise ValueError("campaign must be a safe file name")
    path = ROOT / "runs" / f"{args.campaign}-budget.json"
    if path.exists() and not args.resume:
        raise ValueError("existing campaign requires --resume; spend is never reset")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    selected = args.cases.split(",")
    requests = [(case, ReviseMissionIn.model_validate(data[case])) for case in selected]
    budget = CampaignBudget(Decimal(args.budget), path=path)
    if args.dry_run or not args.live:
        print(
            json.dumps(
                {
                    "cases": selected,
                    "liability_usd": str(budget.liability_usd),
                    "modeled_ceiling_usd": str(budget.limit),
                    "live": False,
                    "note": "No provider calls. Exact next-call reservation includes rendered prompt/schema and maximum output.",
                }
            )
        )
        return True
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write("campaign running")
    adapter = OpenAIChatAdapter.from_settings(get_settings())
    port = BudgetedLLMPort(adapter, budget)
    container = build_container(get_settings(), llm_port=port)
    report_path = ROOT / "runs" / f"{args.campaign}-results.json"
    previous = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if args.resume and report_path.exists()
        else []
    )
    succeeded = True
    reports: list[dict[str, Any]] = []
    try:
        for name, body in requests:
            ledger = UsageLedger(f"revision-eval-{uuid4()}", float(budget.limit))
            ledger.limit_invocations(container.settings.s2_revision_max_calls)
            port.responses.clear()
            start = time.perf_counter()
            report: dict[str, Any] = {"case": name, "success": False}
            try:
                async with asyncio.timeout(container.settings.s2_revision_timeout_seconds):
                    result = await ReviseMissionUseCase(
                        container.llm, container.s2, container.tokens
                    ).execute(body, ledger)
                report.update(success=True, result=result.model_dump(mode="json"))
            except Exception as exc:
                succeeded = False
                report.update(error_type=type(exc).__name__, error=str(exc))
            finally:
                report.update(
                    elapsed_s=round(time.perf_counter() - start, 2),
                    recorded_cost_usd=ledger.total_cost_usd,
                    invocations=[asdict(i) for i in ledger.records],
                    responses=list(port.responses),
                )
                reports.append(report)
                report_path.write_text(
                    json.dumps(previous + reports, ensure_ascii=False, default=str, indent=2),
                    encoding="utf-8",
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in report.items()
                            if k not in ("result", "invocations", "responses")
                        }
                    ),
                    flush=True,
                )
            if not succeeded or budget.state["halted"]:
                break
    finally:
        await container.aclose()
        await adapter.aclose()
        lock_path.unlink()
        print(
            json.dumps(
                {
                    "campaign_liability_usd": str(budget.liability_usd),
                    "unknown_billing": budget.state["halted"],
                }
            )
        )
    return succeeded and len(reports) == len(requests)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "datasets" / "revision_cases.json")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--budget", default="0.50")
    parser.add_argument("--cases", default="A1,A2,A3")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live", action="store_true")
    raise SystemExit(0 if asyncio.run(run(parser.parse_args())) else 1)


if __name__ == "__main__":
    main()
