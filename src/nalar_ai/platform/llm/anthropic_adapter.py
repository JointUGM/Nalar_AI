"""LLMPort over the official Anthropic Python SDK, pointed at SumoPod's Messages endpoint.

This is the only module that imports `anthropic`. If SumoPod turns out to expose only an
OpenAI-style chat endpoint (design doc §12.3), add a second adapter beside this one and
switch it in container.py. Nothing above the port changes.
"""

from typing import Any

import anthropic

from nalar_ai.platform.llm.ports import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    PermanentLLMError,
    TransientLLMError,
    UserBlock,
)
from nalar_ai.settings import Settings
from nalar_ai.shared.errors import ConfigurationError

_EPHEMERAL = {"type": "ephemeral"}
_RETRYABLE = frozenset({408, 409, 429})


def is_retryable_status(status: int) -> bool:
    return status in _RETRYABLE or status >= 500


def _content_block(block: UserBlock) -> dict[str, Any]:
    content: dict[str, Any] = {"type": "text", "text": block.text}
    if block.cache:
        content["cache_control"] = dict(_EPHEMERAL)
    return content


def build_message_params(request: LLMRequest) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "system": [{"type": "text", "text": request.system, "cache_control": dict(_EPHEMERAL)}],
        "messages": [
            {"role": "user", "content": [_content_block(block) for block in request.blocks]}
        ],
    }
    output_config: dict[str, Any] = {}
    if request.json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
    if request.effort is not None:
        output_config["effort"] = request.effort
    if output_config:
        params["output_config"] = output_config
    return params


def parse_message(message: Any) -> LLMResponse:
    usage = message.usage
    parsed_usage = LLMUsage(
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        cache_read_tokens=int(getattr(usage, "cache_read_input_tokens", None) or 0),
        cache_write_tokens=int(getattr(usage, "cache_creation_input_tokens", None) or 0),
    )
    if message.stop_reason == "refusal":
        raise PermanentLLMError("the model refused the request", usage=parsed_usage)
    text = "".join(
        block.text for block in message.content if getattr(block, "type", None) == "text"
    )
    return LLMResponse(
        text=text,
        usage=parsed_usage,
        stop_reason=message.stop_reason,
    )


class AnthropicMessagesAdapter:
    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnthropicMessagesAdapter":
        if not settings.sumopod_messages_base_url:
            # Without a base URL the SDK would call api.anthropic.com with the SumoPod key.
            raise ConfigurationError("NALAR_AI_SUMOPOD_MESSAGES_BASE_URL is not set")
        client = anthropic.AsyncAnthropic(
            api_key=settings.sumopod_api_key.get_secret_value(),
            base_url=settings.sumopod_messages_base_url,
            max_retries=0,  # the gateway owns retries so every attempt is recorded
            timeout=settings.llm_timeout_seconds,
        )
        return cls(client)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            message = await self._client.messages.create(**build_message_params(request))
        except anthropic.APIStatusError as exc:
            detail = f"{exc.status_code}: {exc.message}"
            if is_retryable_status(exc.status_code):
                raise TransientLLMError(detail) from exc
            raise PermanentLLMError(detail) from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise TransientLLMError(f"connection error: {exc}") from exc
        return parse_message(message)

    async def aclose(self) -> None:
        await self._client.close()
