from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from nalar_ai.platform.llm.anthropic_adapter import (
    AnthropicMessagesAdapter,
    build_message_params,
    is_retryable_status,
    parse_message,
)
from nalar_ai.platform.llm.ports import LLMRequest, PermanentLLMError, UserBlock
from nalar_ai.settings import Settings
from nalar_ai.shared.errors import ConfigurationError

SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def _request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "model": "claude-sonnet-5",
        "system": "sys",
        "blocks": (UserBlock("ctx", cache=True), UserBlock("task")),
        "max_tokens": 500,
        "json_schema": SCHEMA,
        "effort": "high",
    }
    base.update(overrides)
    return LLMRequest(**base)


def _message(stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"a": 1}'),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=None,
        ),
        stop_reason=stop_reason,
    )


def test_params_cache_the_system_prompt_and_marked_blocks() -> None:
    params = build_message_params(_request())
    assert params["model"] == "claude-sonnet-5"
    assert params["max_tokens"] == 500
    assert params["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert params["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "task"},
            ],
        }
    ]
    assert params["output_config"] == {
        "format": {"type": "json_schema", "schema": SCHEMA},
        "effort": "high",
    }
    assert "temperature" not in params


def test_params_omit_output_config_when_unused() -> None:
    assert "output_config" not in build_message_params(_request(json_schema=None, effort=None))


def test_parse_message_joins_text_blocks_and_reads_cache_usage() -> None:
    response = parse_message(_message())
    assert response.text == '{"a": 1}'
    assert response.usage.input_tokens == 10
    assert response.usage.cache_read_tokens == 3
    assert response.usage.cache_write_tokens == 0
    assert response.stop_reason == "end_turn"


def test_refusal_is_a_permanent_error() -> None:
    with pytest.raises(PermanentLLMError, match="refused"):
        parse_message(_message(stop_reason="refusal"))


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (404, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (529, True),
    ],
)
def test_retryable_statuses(status: int, retryable: bool) -> None:
    assert is_retryable_status(status) is retryable


class _StubMessages:
    def __init__(self, message: SimpleNamespace) -> None:
        self.message = message
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.message


async def test_adapter_sends_built_params_through_the_sdk() -> None:
    stub = _StubMessages(_message())
    adapter = AnthropicMessagesAdapter(SimpleNamespace(messages=stub))
    response = await adapter.complete(_request())
    assert response.text == '{"a": 1}'
    assert stub.calls[0] == build_message_params(_request())


def test_from_settings_refuses_an_empty_base_url() -> None:
    settings = Settings.model_construct(service_key=SecretStr("k"), sumopod_messages_base_url="")
    with pytest.raises(ConfigurationError, match="SUMOPOD_MESSAGES_BASE_URL"):
        AnthropicMessagesAdapter.from_settings(settings)
