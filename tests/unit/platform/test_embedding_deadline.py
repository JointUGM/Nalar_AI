import asyncio
from collections.abc import Sequence

import pytest

from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.embeddings.ports import EmbeddingBatch, TransientEmbeddingError
from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.shared.enums import CallStatus
from nalar_ai.shared.errors import UpstreamUnavailableError
from nalar_ai.shared.provenance import UsageLedger

MODEL = "text-embedding-3-small"


class StuckEmbedder:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        self.started.set()
        await asyncio.Event().wait()  # never returns
        raise AssertionError("unreachable")


class FlakyEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        self.calls += 1
        if self.calls == 1:
            raise TransientEmbeddingError("503")
        return await HashingEmbedder().embed(texts, model=model, dimensions=dimensions)


async def test_a_stuck_call_hits_its_deadline_and_is_recorded() -> None:
    service = EmbeddingService(StuckEmbedder(), model=MODEL, dimensions=8)
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(UpstreamUnavailableError):
        await service.embed(["gaya"], tag="query", ledger=ledger, timeout_s=0.05, max_attempts=1)
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    assert record.error_message == "deadline exceeded after 0.05s"


async def test_max_attempts_overrides_the_service_default() -> None:
    flaky = FlakyEmbedder()
    service = EmbeddingService(flaky, model=MODEL, dimensions=8)
    with pytest.raises(UpstreamUnavailableError):
        await service.embed(["gaya"], tag="query", ledger=UsageLedger("r", 1.0), max_attempts=1)
    assert flaky.calls == 1


async def test_default_deadline_bounds_calls_without_an_override() -> None:
    service = EmbeddingService(
        StuckEmbedder(), model=MODEL, dimensions=8, default_timeout_s=0.01, max_attempts=1
    )
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(UpstreamUnavailableError):
        await service.embed(["gaya"], tag="query", ledger=ledger)
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    assert record.error_message == "deadline exceeded after 0.01s"


async def test_caller_cancellation_preserves_the_embedding_attempt() -> None:
    port = StuckEmbedder()
    service = EmbeddingService(port, model=MODEL, dimensions=8)
    ledger = UsageLedger("r", 1.0)
    task = asyncio.create_task(service.embed(["gaya"], tag="query", ledger=ledger))
    await port.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    assert record.error_message == "embedding call cancelled"
