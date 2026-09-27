import uuid

import pytest

from nalar_ai.shared.enums import AiPurpose, CallStatus, RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import BudgetExceededError
from nalar_ai.shared.provenance import InvocationRecord, RetrievalRef, UsageLedger


def _record(cost: float) -> InvocationRecord:
    return InvocationRecord(
        purpose=AiPurpose.KB_EXTRACT,
        model="claude-sonnet-5",
        prompt_version="s1.extract_concepts@v1",
        status=CallStatus.SUCCESS,
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        latency_ms=12,
        cost_usd=cost,
        request_id="req-1",
        retrieval=(
            RetrievalRef(RetrievalSource.MATERIAL_CHUNK, uuid.uuid4(), RetrievalPath.LINK, 1),
        ),
    )


def test_ledger_sums_cost_and_keeps_order() -> None:
    ledger = UsageLedger(request_id="req-1", max_cost_usd=1.0)
    ledger.add(_record(0.25))
    ledger.add(_record(0.5))
    assert ledger.total_cost_usd == pytest.approx(0.75)
    assert [r.cost_usd for r in ledger.records] == [0.25, 0.5]


def test_ledger_blocks_calls_once_the_budget_is_spent() -> None:
    ledger = UsageLedger(request_id="req-1", max_cost_usd=0.5)
    ledger.ensure_budget()
    ledger.add(_record(0.5))
    with pytest.raises(BudgetExceededError) as info:
        ledger.ensure_budget()
    assert info.value.code == "budget_exceeded"
    assert info.value.details["limit_usd"] == 0.5
