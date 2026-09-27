from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class UserBlock:
    text: str
    cache: bool = False


@dataclass(frozen=True, slots=True)
class LLMRequest:
    model: str
    system: str
    blocks: tuple[UserBlock, ...]
    max_tokens: int
    json_schema: dict[str, Any] | None = None
    effort: str | None = None


@dataclass(frozen=True, slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    usage: LLMUsage
    stop_reason: str | None = None


class LLMPort(Protocol):
    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class TransientLLMError(Exception):
    """Retryable: timeouts, rate limits, overload, 5xx, connection failures."""


class PermanentLLMError(Exception):
    """Not retryable: bad request, auth, refusal."""
