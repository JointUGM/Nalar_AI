from collections.abc import Sequence

import httpx

from nalar_ai.platform.embeddings.ports import (
    EmbeddingBatch,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from nalar_ai.settings import Settings
from nalar_ai.shared.errors import ConfigurationError

_RETRYABLE = frozenset({408, 409, 429})


class OpenAICompatEmbedder:
    """SumoPod's OpenAI-compatible /embeddings endpoint (text-embedding-3-small)."""

    def __init__(self, client: httpx.AsyncClient, *, base_url: str, api_key: str) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._api_key = api_key

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatEmbedder":
        if not settings.sumopod_openai_base_url:
            raise ConfigurationError("NALAR_AI_SUMOPOD_OPENAI_BASE_URL is not set")
        return cls(
            httpx.AsyncClient(timeout=settings.llm_timeout_seconds),
            base_url=settings.sumopod_openai_base_url,
            api_key=settings.sumopod_api_key.get_secret_value(),
        )

    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        try:
            response = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": model,
                    "input": list(texts),
                    "dimensions": dimensions,
                    "encoding_format": "float",
                },
            )
        except httpx.TransportError as exc:  # includes timeouts
            raise TransientEmbeddingError(f"connection error: {exc}") from exc
        status = response.status_code
        if status in _RETRYABLE or status >= 500:
            raise TransientEmbeddingError(f"{status}: {response.text[:200]}")
        if status >= 400:
            raise PermanentEmbeddingError(f"{status}: {response.text[:200]}")
        try:
            payload = response.json()
            returned_model = payload.get("model")
            if returned_model is not None and returned_model != model:
                raise PermanentEmbeddingError("embedding response model does not match request")
            data = payload["data"]
            indices = [row["index"] for row in data]
            if any(type(index) is not int for index in indices) or sorted(indices) != list(
                range(len(texts))
            ):
                raise PermanentEmbeddingError(
                    "embedding indices must match every input exactly once"
                )
            rows = sorted(data, key=lambda row: row["index"])
            return EmbeddingBatch(
                vectors=[[float(value) for value in row["embedding"]] for row in rows],
                total_tokens=int(payload.get("usage", {}).get("total_tokens", 0)),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise PermanentEmbeddingError(f"malformed embeddings response: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()
