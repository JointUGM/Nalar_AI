import json
from collections.abc import Sequence

import httpx
import pytest
import respx

from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.embeddings.openai_compat import OpenAICompatEmbedder
from nalar_ai.platform.embeddings.ports import (
    EmbeddingBatch,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.shared.enums import AiPurpose, CallStatus
from nalar_ai.shared.errors import (
    InvalidInputError,
    ModelCallRejectedError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.vectors import cosine

BASE = "https://sumopod.test/v1"
MODEL = "text-embedding-3-small"


@respx.mock
async def test_openai_compat_sorts_by_index_and_reads_usage() -> None:
    route = respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"total_tokens": 7},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        embedder = OpenAICompatEmbedder(client, base_url=BASE, api_key="k")
        batch = await embedder.embed(["a", "b"], model=MODEL, dimensions=2)
    assert batch.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert batch.total_tokens == 7
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer k"
    assert json.loads(request.content) == {
        "model": MODEL,
        "input": ["a", "b"],
        "dimensions": 2,
        "encoding_format": "float",
    }


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [-1, 0], [0.5, 1], [False, 1], ["0", 1]])
@respx.mock
async def test_openai_compat_rejects_invalid_embedding_indices(indices: list[object]) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"index": index, "embedding": [1.0, 0.0]} for index in indices],
                "usage": {"total_tokens": 7},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        embedder = OpenAICompatEmbedder(client, base_url=BASE, api_key="k")
        with pytest.raises(PermanentEmbeddingError, match="indices"):
            await embedder.embed(["a", "b"], model=MODEL, dimensions=2)


@respx.mock
async def test_openai_compat_rejects_vectors_from_a_different_model() -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "different-model",
                "data": [{"index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"total_tokens": 7},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        embedder = OpenAICompatEmbedder(client, base_url=BASE, api_key="k")
        with pytest.raises(PermanentEmbeddingError, match="model"):
            await embedder.embed(["a"], model=MODEL, dimensions=2)


@respx.mock
async def test_openai_compat_maps_status_codes() -> None:
    route = respx.post(f"{BASE}/embeddings")
    async with httpx.AsyncClient() as client:
        embedder = OpenAICompatEmbedder(client, base_url=BASE, api_key="k")
        route.mock(return_value=httpx.Response(429, json={}))
        with pytest.raises(TransientEmbeddingError):
            await embedder.embed(["a"], model=MODEL, dimensions=2)
        route.mock(return_value=httpx.Response(400, text="bad"))
        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed(["a"], model=MODEL, dimensions=2)


def test_hashing_embedder_is_deterministic_and_similarity_follows_shared_words() -> None:
    a = HashingEmbedder.vector("gaya gesek menghambat gerak", 1536)
    assert a == HashingEmbedder.vector("gaya gesek menghambat gerak", 1536)
    related = HashingEmbedder.vector("gaya gesek pada benda", 1536)
    unrelated = HashingEmbedder.vector("tekanan zat cair", 1536)
    assert cosine(a, related) > cosine(a, unrelated)


async def test_service_batches_records_and_returns_tuples() -> None:
    service = EmbeddingService(HashingEmbedder(), model=MODEL, dimensions=1536, batch_size=2)
    ledger = UsageLedger("req-1", 1.0)
    vectors = await service.embed(["a b", "c", "d e f", "g", "h"], tag="chunk", ledger=ledger)
    assert len(vectors) == 5
    assert all(isinstance(v, tuple) and len(v) == 1536 for v in vectors)
    assert [r.prompt_version for r in ledger.records] == ["embed.chunk"] * 3
    assert all(r.purpose is AiPurpose.EMBEDDING for r in ledger.records)
    assert ledger.records[0].input_tokens == 3
    assert ledger.records[0].request_id == "req-1"


async def test_empty_input_needs_no_call_and_blank_text_is_rejected() -> None:
    service = EmbeddingService(HashingEmbedder(), model=MODEL, dimensions=1536)
    ledger = UsageLedger("r", 1.0)
    assert await service.embed([], tag="chunk", ledger=ledger) == []
    with pytest.raises(InvalidInputError):
        await service.embed(["ok", "  "], tag="chunk", ledger=ledger)
    assert ledger.records == ()


class _WrongDimensions:
    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        return EmbeddingBatch(vectors=[[1.0] for _ in texts], total_tokens=1)


async def test_wrong_dimensions_are_rejected() -> None:
    service = EmbeddingService(_WrongDimensions(), model=MODEL, dimensions=1536)
    with pytest.raises(OutputValidationError, match="dimension"):
        await service.embed(["a"], tag="chunk", ledger=UsageLedger("r", 1.0))


class _Flaky:
    def __init__(self, failures: int) -> None:
        self.failures = failures

    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        if self.failures > 0:
            self.failures -= 1
            raise TransientEmbeddingError("503")
        return await HashingEmbedder().embed(texts, model=model, dimensions=dimensions)


async def _no_sleep(_: float) -> None:
    return None


async def test_transient_errors_are_retried_and_recorded() -> None:
    service = EmbeddingService(_Flaky(2), model=MODEL, dimensions=1536, sleep=_no_sleep)
    ledger = UsageLedger("r", 1.0)
    assert len(await service.embed(["a"], tag="query", ledger=ledger)) == 1
    assert [r.status for r in ledger.records] == [
        CallStatus.ERROR,
        CallStatus.ERROR,
        CallStatus.SUCCESS,
    ]


async def test_exhausted_retries_raise_upstream_unavailable() -> None:
    service = EmbeddingService(_Flaky(9), model=MODEL, dimensions=1536, sleep=_no_sleep)
    with pytest.raises(UpstreamUnavailableError):
        await service.embed(["a"], tag="query", ledger=UsageLedger("r", 1.0))


@respx.mock
async def test_openai_compat_rejects_a_malformed_success_payload() -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(200, text="<html>oops</html>")
    )
    async with httpx.AsyncClient() as client:
        embedder = OpenAICompatEmbedder(client, base_url=BASE, api_key="k")
        with pytest.raises(PermanentEmbeddingError, match="malformed"):
            await embedder.embed(["a"], model=MODEL, dimensions=2)


class _Broken:
    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        raise ValueError("unexpected shape")


async def test_unexpected_embedding_errors_are_recorded_and_rejected() -> None:
    service = EmbeddingService(_Broken(), model=MODEL, dimensions=1536)
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(ModelCallRejectedError):
        await service.embed(["a"], tag="chunk", ledger=ledger)
    assert [r.status for r in ledger.records] == [CallStatus.ERROR]
