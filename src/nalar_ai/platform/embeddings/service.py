"""The one path for embeddings: batching, retries, validation and provenance."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence

from nalar_ai.platform.embeddings.ports import (
    EmbeddingBatch,
    EmbeddingPort,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from nalar_ai.platform.llm.pricing import embedding_cost_usd
from nalar_ai.platform.retry import retry_transient
from nalar_ai.shared.enums import AiPurpose, CallStatus
from nalar_ai.shared.errors import (
    InvalidInputError,
    ModelCallRejectedError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import InvocationRecord, UsageLedger

Vector = tuple[float, ...]


class EmbeddingService:
    def __init__(
        self,
        port: EmbeddingPort,
        *,
        model: str,
        dimensions: int,
        batch_size: int = 256,
        max_attempts: int = 3,
        base_delay_s: float = 0.5,
        provider: str = "sumopod",
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        embedding_cost_usd(model, 0)  # fail at startup if the model has no price
        self._port = port
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._provider = provider
        self._clock = clock
        self._sleep = sleep

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str], *, tag: str, ledger: UsageLedger) -> list[Vector]:
        if any(not text.strip() for text in texts):
            raise InvalidInputError("cannot embed empty text")
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch, tag, ledger))
        return vectors

    async def _embed_batch(self, batch: list[str], tag: str, ledger: UsageLedger) -> list[Vector]:
        async def attempt() -> EmbeddingBatch:
            ledger.ensure_budget()
            started = self._clock()
            try:
                result = await self._port.embed(
                    batch, model=self._model, dimensions=self._dimensions
                )
            except (TransientEmbeddingError, PermanentEmbeddingError) as exc:
                ledger.add(self._record(tag, 0, started, ledger, str(exc)))
                raise
            except Exception as exc:  # an odd payload must not drop the attempt's record
                detail = f"unexpected embedding error: {exc!r}"
                ledger.add(self._record(tag, 0, started, ledger, detail))
                raise PermanentEmbeddingError(detail) from exc
            ledger.add(self._record(tag, result.total_tokens, started, ledger))
            return result

        try:
            result = await retry_transient(
                attempt,
                is_transient=lambda exc: isinstance(exc, TransientEmbeddingError),
                max_attempts=self._max_attempts,
                base_delay_s=self._base_delay_s,
                sleep=self._sleep,
            )
        except TransientEmbeddingError as exc:
            raise UpstreamUnavailableError(f"embedding provider unavailable: {exc}") from exc
        except PermanentEmbeddingError as exc:
            raise ModelCallRejectedError(f"embedding call rejected: {exc}") from exc
        return self._validated(result, len(batch))

    def _validated(self, result: EmbeddingBatch, expected: int) -> list[Vector]:
        if len(result.vectors) != expected:
            raise OutputValidationError(
                f"expected {expected} embeddings, got {len(result.vectors)}"
            )
        for vector in result.vectors:
            if len(vector) != self._dimensions:
                raise OutputValidationError(
                    f"embedding dimension {len(vector)} != configured {self._dimensions}"
                )
            if not all(math.isfinite(value) for value in vector):
                raise OutputValidationError("embedding contains non-finite values")
        return [tuple(vector) for vector in result.vectors]

    def _record(
        self, tag: str, tokens: int, started: float, ledger: UsageLedger, error: str | None = None
    ) -> InvocationRecord:
        return InvocationRecord(
            purpose=AiPurpose.EMBEDDING,
            model=self._model,
            prompt_version=f"embed.{tag}",
            status=CallStatus.ERROR if error else CallStatus.SUCCESS,
            input_tokens=tokens,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            latency_ms=int((self._clock() - started) * 1000),
            cost_usd=round(embedding_cost_usd(self._model, tokens), 6),
            request_id=ledger.request_id,
            provider=self._provider,
            error_message=error,
        )
