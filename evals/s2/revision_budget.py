import asyncio
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from nalar_ai.platform.llm.ports import LLMPort, LLMRequest, LLMResponse, PermanentLLMError
from nalar_ai.platform.llm.pricing import LLM_PRICES, llm_cost_usd


class EvalBudgetLimit(PermanentLLMError):
    pass


class CampaignBudget:
    def __init__(
        self, limit_usd: Decimal, margin: Decimal = Decimal("0.10"), path: Path | None = None
    ) -> None:
        if not Decimal("0") < limit_usd <= Decimal("1"):
            raise ValueError("campaign budget must be positive and at most US$1")
        self.limit = limit_usd * (1 - margin)
        self.path = path
        self.state: dict[str, Any] = {"billed": "0", "reservations": {}, "halted": False}
        if path is not None and path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))

    @property
    def liability_usd(self) -> Decimal:
        return Decimal(self.state["billed"]) + sum(
            (Decimal(v) for v in self.state["reservations"].values()), Decimal(0)
        )

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state), encoding="utf-8")
            temporary.replace(self.path)

    def reserve(self, amount: Decimal) -> str:
        if self.state["halted"] or self.liability_usd + amount > self.limit:
            raise EvalBudgetLimit("eval campaign budget exhausted or billing unknown")
        from uuid import uuid4

        key = str(uuid4())
        self.state["reservations"][key] = str(amount)
        self.save()
        return key

    def record_known_charge(self, amount: Decimal) -> None:
        self.state["billed"] = str(Decimal(self.state["billed"]) + amount)
        self.save()

    def reconcile(self, key: str, cost: Decimal) -> None:
        reserved = Decimal(self.state["reservations"].pop(key))
        self.record_known_charge(cost)
        if cost > reserved:
            self.state["halted"] = True
        self.save()

    def unknown_charge(self) -> None:
        self.state["halted"] = True
        self.save()


def reservation(request: LLMRequest) -> Decimal:
    price = LLM_PRICES[request.model]
    # Byte-count allowance deliberately overestimates normal tokenization and includes schema.
    byte_count = len(json.dumps(asdict(request), ensure_ascii=False).encode("utf-8")) + 2048
    input_price = Decimal(str(price.input_per_mtok)) * max(
        Decimal(1), Decimal(str(price.cache_write_multiplier))
    )
    return (
        Decimal(byte_count) * input_price
        + Decimal(request.max_tokens) * Decimal(str(price.output_per_mtok))
    ) / Decimal(1_000_000)


class BudgetedLLMPort:
    def __init__(self, port: LLMPort, budget: CampaignBudget) -> None:
        self.port, self.budget = port, budget
        self.lock = asyncio.Lock()
        self.responses: list[dict[str, str]] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        async with self.lock:
            key = self.budget.reserve(reservation(request))
            try:
                result = await self.port.complete(request)
            except PermanentLLMError as exc:
                if any(
                    (
                        exc.usage.input_tokens,
                        exc.usage.output_tokens,
                        exc.usage.cache_read_tokens,
                        exc.usage.cache_write_tokens,
                    )
                ):
                    self.budget.reconcile(key, Decimal(str(llm_cost_usd(request.model, exc.usage))))
                else:
                    self.budget.unknown_charge()
                raise
            except BaseException:
                self.budget.unknown_charge()
                raise
            self.responses.append({"model": request.model, "response": result.text})
            self.budget.reconcile(key, Decimal(str(llm_cost_usd(request.model, result.usage))))
            return result
