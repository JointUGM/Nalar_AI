from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    total_tokens: int


class EmbeddingPort(Protocol):
    async def embed(
        self, texts: Sequence[str], *, model: str, dimensions: int
    ) -> EmbeddingBatch: ...


class TransientEmbeddingError(Exception):
    """Retryable: timeouts, rate limits, 5xx, connection failures."""


class PermanentEmbeddingError(Exception):
    """Not retryable: bad request, auth, unknown model."""
