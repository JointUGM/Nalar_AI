import json
from typing import Any, Literal

import httpx
import pytest
from pydantic import SecretStr

from nalar_ai.container import _default_llm_port
from nalar_ai.platform.llm.anthropic_adapter import AnthropicMessagesAdapter
from nalar_ai.platform.llm.openai_chat_adapter import (
    OpenAIChatAdapter,
    build_chat_body,
    normalize_arguments,
    parse_chat_completion,
)
from nalar_ai.platform.llm.ports import (
    LLMRequest,
    PermanentLLMError,
    TransientLLMError,
    UserBlock,
)
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


def _completion(
    *, finish_reason: str = "stop", content: str | None = '{"a": 1}', **usage: int
) -> dict[str, Any]:
    # LiteLLM's shape: prompt_tokens INCLUDES cache reads and writes (probe P6).
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {
            "prompt_tokens": 4938,
            "completion_tokens": 7,
            "cache_read_input_tokens": 4924,
            "cache_creation_input_tokens": 0,
            **usage,
        },
    }


def test_body_caches_the_system_prompt_and_marked_blocks() -> None:
    body = build_chat_body(_request())
    assert body["model"] == "claude-sonnet-5"
    assert body["max_tokens"] == 500
    assert body["messages"] == [
        {
            "role": "system",
            "content": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ctx", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "task"},
            ],
        },
    ]
    # SumoPod drops response_format silently (DECISIONS P8); a forced tool call works.
    assert body["tools"] == [
        {
            "type": "function",
            "function": {"name": "output", "strict": True, "parameters": SCHEMA},
        }
    ]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "output"}}
    assert "response_format" not in body
    assert body["output_config"] == {"effort": "high"}
    assert "temperature" not in body


def test_body_omits_unused_options() -> None:
    body = build_chat_body(_request(json_schema=None, effort=None))
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "output_config" not in body


def test_parse_subtracts_cached_tokens_from_input() -> None:
    response = parse_chat_completion(_completion())
    assert response.text == '{"a": 1}'
    assert response.usage.input_tokens == 14
    assert response.usage.output_tokens == 7
    assert response.usage.cache_read_tokens == 4924
    assert response.usage.cache_write_tokens == 0
    assert response.stop_reason == "stop"


def test_parse_returns_the_forced_tool_arguments_as_text() -> None:
    payload = _completion(finish_reason="tool_calls", content=None)
    payload["choices"][0]["message"]["tool_calls"] = [
        {"function": {"name": "output", "arguments": '{"b": 2}'}}
    ]
    response = parse_chat_completion(payload)
    assert response.text == '{"b": 2}'
    assert response.stop_reason == "tool_calls"


def test_parse_reads_a_cache_write() -> None:
    usage = parse_chat_completion(
        _completion(cache_read_input_tokens=0, cache_creation_input_tokens=4924)
    ).usage
    assert (usage.input_tokens, usage.cache_write_tokens) == (14, 4924)


def test_length_maps_to_max_tokens_for_the_gateway() -> None:
    assert parse_chat_completion(_completion(finish_reason="length")).stop_reason == "max_tokens"


@pytest.mark.parametrize("reason", ["content_filter", "refusal"])
def test_refusal_is_a_permanent_error(reason: str) -> None:
    with pytest.raises(PermanentLLMError, match="refused"):
        parse_chat_completion(_completion(finish_reason=reason))


def test_malformed_response_is_a_permanent_error() -> None:
    with pytest.raises(PermanentLLMError, match="malformed"):
        parse_chat_completion({"choices": []})


def _adapter(handler: Any) -> OpenAIChatAdapter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIChatAdapter(client, base_url="https://sumopod.test/v1/", api_key="k")


async def test_adapter_posts_the_body_with_a_bearer_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion())

    response = await _adapter(handler).complete(_request())
    assert response.text == '{"a": 1}'
    assert str(seen[0].url) == "https://sumopod.test/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer k"
    assert json.loads(seen[0].content) == build_chat_body(_request())


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (400, PermanentLLMError),
        (401, PermanentLLMError),
        (404, PermanentLLMError),
        (408, TransientLLMError),
        (409, TransientLLMError),
        (429, TransientLLMError),
        (500, TransientLLMError),
        (529, TransientLLMError),
    ],
)
async def test_status_errors_are_classified(status: int, error: type[Exception]) -> None:
    adapter = _adapter(lambda _: httpx.Response(status, text="nope"))
    with pytest.raises(error, match=str(status)):
        await adapter.complete(_request())


async def test_connection_failure_is_transient() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(TransientLLMError, match="connection"):
        await _adapter(handler).complete(_request())


def test_from_settings_refuses_an_empty_base_url() -> None:
    settings = Settings.model_construct(service_key=SecretStr("k"), sumopod_openai_base_url="")
    with pytest.raises(ConfigurationError, match="SUMOPOD_OPENAI_BASE_URL"):
        OpenAIChatAdapter.from_settings(settings)


@pytest.mark.parametrize(
    ("api", "adapter_type"),
    [("chat", OpenAIChatAdapter), ("messages", AnthropicMessagesAdapter)],
)
def test_container_picks_the_adapter_from_settings(
    api: Literal["chat", "messages"], adapter_type: type
) -> None:
    settings = Settings.model_construct(
        service_key=SecretStr("k"),
        sumopod_api=api,
        sumopod_messages_base_url="https://sumopod.test/anthropic",
        sumopod_openai_base_url="https://sumopod.test/v1",
    )
    closers: list[Any] = []
    assert isinstance(_default_llm_port(settings, closers), adapter_type)
    assert len(closers) == 1


def test_chat_is_the_default_api() -> None:
    assert Settings.model_fields["sumopod_api"].default == "chat"


NULLABLE = {"anyOf": [{"type": "string"}, {"type": "null"}]}
NESTED_SCHEMA = {
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "note": NULLABLE},
            "required": ["name", "note"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
        "extra": NULLABLE,
        "label": {"type": "string"},
    },
    "required": ["items", "extra", "label"],
    "additionalProperties": False,
}


def test_fill_missing_nulls_adds_only_omitted_nullable_fields() -> None:
    # SumoPod ignores strict tool schemas, and Claude omits null fields (DECISIONS P10).
    value = {"items": [{"name": "a"}, {"name": "b", "note": "x"}]}
    assert normalize_arguments(value, NESTED_SCHEMA) == {
        "items": [{"name": "a", "note": None}, {"name": "b", "note": "x"}],
        "extra": None,  # nullable: filled
        # "label" is required but not nullable: left missing, so validation still fails
    }


async def test_adapter_fills_omitted_nulls_in_structured_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = _completion(finish_reason="tool_calls", content=None)
        payload["choices"][0]["message"]["tool_calls"] = [
            {"function": {"name": "output", "arguments": '{"items": [], "label": "ok"}'}}
        ]
        return httpx.Response(200, json=payload)

    response = await _adapter(handler).complete(_request(json_schema=NESTED_SCHEMA))
    assert json.loads(response.text) == {"items": [], "label": "ok", "extra": None}


def test_normalize_decodes_arrays_and_objects_sent_as_json_strings() -> None:
    # Claude via SumoPod sometimes JSON-encodes a nested field (DECISIONS P13).
    value = {"items": '[{"name": "a"}]', "label": "[not decoded]", "extra": "[x]"}
    assert normalize_arguments(value, NESTED_SCHEMA) == {
        "items": [{"name": "a", "note": None}],
        "label": "[not decoded]",  # a string field stays a string
        "extra": "[x]",
    }


def test_normalize_leaves_undecodable_strings_for_validation_to_reject() -> None:
    assert normalize_arguments({"items": "[oops"}, NESTED_SCHEMA)["items"] == "[oops"


def test_normalize_unwraps_only_redundant_array_field_wrappers() -> None:
    assert normalize_arguments({"items": {"items": [{"name": "a"}]}}, NESTED_SCHEMA)["items"] == [
        {"name": "a", "note": None}
    ]
    invalid_values: tuple[dict[str, Any], ...] = (
        {"other": []},
        {"items": "invalid"},
        {"items": [], "extra": 1},
    )
    for invalid in invalid_values:
        assert normalize_arguments({"items": invalid}, NESTED_SCHEMA)["items"] == invalid
    schema = {
        "type": "object",
        "properties": {"box": {"type": "object", "properties": {"box": {"type": "array"}}}},
    }
    assert normalize_arguments({"box": {"box": []}}, schema) == {"box": {"box": []}}


def test_normalize_unwraps_arguments_nested_under_one_unknown_key() -> None:
    # Claude via SumoPod sometimes nests the whole answer under "$PARAMETER_NAME" (P13).
    wrapped = {"$PARAMETER_NAME": {"items": [], "label": "ok"}}
    assert normalize_arguments(wrapped, NESTED_SCHEMA) == {
        "items": [],
        "label": "ok",
        "extra": None,
    }
    single: dict[str, Any] = {"label": {"items": []}}  # a real property is never unwrapped
    assert normalize_arguments(single, NESTED_SCHEMA)["label"] == {"items": []}


async def test_adapter_normalizes_a_wrapped_and_stringified_tool_call() -> None:
    arguments = json.dumps({"$PARAMETER_NAME": {"items": "[]", "label": "ok"}})

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _completion(finish_reason="tool_calls", content=None)
        payload["choices"][0]["message"]["tool_calls"] = [
            {"function": {"name": "output", "arguments": arguments}}
        ]
        return httpx.Response(200, json=payload)

    response = await _adapter(handler).complete(_request(json_schema=NESTED_SCHEMA))
    assert json.loads(response.text) == {"items": [], "label": "ok", "extra": None}


async def test_adapter_leaves_invalid_json_for_the_gateway_to_report() -> None:
    adapter = _adapter(lambda _: httpx.Response(200, json=_completion(content="not json")))
    assert (await adapter.complete(_request())).text == "not json"


def test_non_claude_models_get_no_cache_marker_or_effort() -> None:
    # Some OpenAI models reject cache_control with a 400 (DECISIONS M2); effort is Claude-only.
    body = build_chat_body(_request(model="gpt-5.4-mini"))
    parts = [part for message in body["messages"] for part in message["content"]]
    assert all("cache_control" not in part for part in parts)
    assert "output_config" not in body
    assert body["tool_choice"] == {"type": "function", "function": {"name": "output"}}


def test_model_params_are_merged_into_the_body() -> None:
    body = build_chat_body(_request(model="qwen3.8-max"), {"enable_thinking": False})
    assert body["enable_thinking"] is False


async def test_adapter_applies_params_for_that_model_only() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAIChatAdapter(
        client,
        base_url="https://sumopod.test/v1",
        api_key="k",
        model_params={"qwen3.8-max": {"enable_thinking": False}},
    )
    await adapter.complete(_request(model="qwen3.8-max", json_schema=None))
    await adapter.complete(_request(model="gpt-5.4-mini", json_schema=None))
    assert seen[0]["enable_thinking"] is False
    assert "enable_thinking" not in seen[1]


def test_default_settings_turn_thinking_off_for_the_qwen_judge() -> None:
    # qwen refuses a forced tool call while thinking (DECISIONS M1).
    params = Settings.model_fields["llm_model_params"].default
    assert params["qwen3.8-max"] == {"enable_thinking": False}
