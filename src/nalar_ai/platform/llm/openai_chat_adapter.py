"""LLMPort over SumoPod's OpenAI-style /chat/completions endpoint (a LiteLLM proxy).

SumoPod's Anthropic Messages route answers 404 for every Claude model (probe of
2026-09-28, DECISIONS P2), so this is the default adapter. Claude features still pass
through LiteLLM: `cache_control` on content parts, and structured output as a forced
tool call.
"""

import json
from collections.abc import Mapping
from typing import Any

import httpx

from nalar_ai.platform.llm.anthropic_adapter import is_retryable_status
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
_REFUSALS = frozenset({"content_filter", "refusal"})
_TOOL = "output"


def _is_claude(model: str) -> bool:
    return model.startswith("claude")


def _part(block: UserBlock, *, claude: bool) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "text", "text": block.text}
    if block.cache and claude:
        # Claude-only: some OpenAI models reject it with a 400 (DECISIONS M2); the other
        # providers cache repeated prefixes on their own.
        part["cache_control"] = dict(_EPHEMERAL)
    return part


def build_chat_body(request: LLMRequest, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """`params` are extra request fields for this model (settings.llm_model_params)."""
    claude = _is_claude(request.model)
    system = UserBlock(request.system, cache=True)
    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": [
            {"role": "system", "content": [_part(system, claude=claude)]},
            {"role": "user", "content": [_part(block, claude=claude) for block in request.blocks]},
        ],
    }
    if request.json_schema is not None:
        # SumoPod drops response_format silently (DECISIONS P8); a forced tool call is
        # honoured, and the gateway still validates the arguments against the schema.
        body["tools"] = [
            {
                "type": "function",
                "function": {"name": _TOOL, "strict": True, "parameters": request.json_schema},
            }
        ]
        body["tool_choice"] = {"type": "function", "function": {"name": _TOOL}}
    if request.effort is not None and claude:
        # ponytail: SumoPod currently drops this silently (Haiku accepts it; DECISIONS P7).
        # Sent in the native shape so it takes effect if the proxy starts passing it on.
        body["output_config"] = {"effort": request.effort}
    body.update(params or {})
    return body


def parse_chat_completion(payload: dict[str, Any]) -> LLMResponse:
    try:
        choice = payload["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = choice["message"]
        tool_calls = message.get("tool_calls") or []
        text = tool_calls[0]["function"]["arguments"] if tool_calls else message.get("content")
        text = text or ""
        usage = payload.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        # LiteLLM's prompt_tokens includes cached tokens (DECISIONS P6); the pricing
        # table bills input, cache reads and cache writes separately.
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        raise PermanentLLMError(f"malformed chat completion: {exc!r}") from exc
    parsed_usage = LLMUsage(
        input_tokens=max(prompt_tokens - cache_read - cache_write, 0),
        output_tokens=output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
    if finish_reason in _REFUSALS or message.get("refusal"):
        raise PermanentLLMError("the model refused the request", usage=parsed_usage)
    return LLMResponse(
        text=text,
        usage=parsed_usage,
        stop_reason="max_tokens" if finish_reason == "length" else finish_reason,
    )


def _allows_null(schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    return (
        kind == "null"
        or (isinstance(kind, list) and "null" in kind)
        or any(branch.get("type") == "null" for branch in schema.get("anyOf", ()))
    )


_JSON_KINDS: dict[str, type] = {"array": list, "object": dict}


def normalize_arguments(value: Any, schema: dict[str, Any]) -> Any:
    """Undo Claude's tool-argument quirks, which SumoPod passes on because it does not enforce
    strict tool schemas (DECISIONS P10, P13):

    - the whole answer nested under one unknown key such as "$PARAMETER_NAME": unwrapped;
    - an array or object field sent as a JSON string: decoded;
    - an array nested under a redundant copy of its own field name: unwrapped;
    - a required-but-nullable field left out: filled with null.

    Anything else is left as it is, so validation still rejects it.
    """
    if isinstance(value, dict) and len(value) == 1:
        ((key, inner),) = value.items()
        if key not in schema.get("properties", {}) and isinstance(inner, dict):
            value = inner
    return _normalize(value, schema, schema)


def _normalize(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> Any:
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        schema = root.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})
    kind = _JSON_KINDS.get(schema.get("type", ""))
    if isinstance(value, str) and kind is not None:
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, kind):
            value = decoded
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", ()):
            if name not in value and _allows_null(properties.get(name, {})):
                value[name] = None
        for name, sub in properties.items():
            if name in value:
                field_schema = sub
                field_ref = sub.get("$ref")
                if isinstance(field_ref, str) and field_ref.startswith("#/$defs/"):
                    field_schema = root.get("$defs", {}).get(field_ref.removeprefix("#/$defs/"), {})
                field = value[name]
                if (
                    field_schema.get("type") == "array"
                    and isinstance(field, dict)
                    and list(field) == [name]
                    and isinstance(field[name], list)
                ):
                    value[name] = field[name]
                value[name] = _normalize(value[name], sub, root)
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [_normalize(item, schema["items"], root) for item in value]
    for branch in schema.get("anyOf", ()):
        if branch.get("type") != "null":
            value = _normalize(value, branch, root)
    return value


class OpenAIChatAdapter:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        api_key: str,
        model_params: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._client = client
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model_params = dict(model_params or {})

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIChatAdapter":
        if not settings.sumopod_openai_base_url:
            raise ConfigurationError("NALAR_AI_SUMOPOD_OPENAI_BASE_URL is not set")
        return cls(
            httpx.AsyncClient(timeout=settings.llm_timeout_seconds),
            base_url=settings.sumopod_openai_base_url,
            api_key=settings.sumopod_api_key.get_secret_value(),
            model_params=settings.llm_model_params,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = await self._client.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=build_chat_body(request, self._model_params.get(request.model)),
            )
        except httpx.TransportError as exc:  # includes timeouts
            raise TransientLLMError(f"connection error: {exc}") from exc
        status = response.status_code
        if status >= 400:
            detail = f"{status}: {response.text[:200]}"
            if is_retryable_status(status):
                raise TransientLLMError(detail)
            raise PermanentLLMError(detail)
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentLLMError(f"malformed chat completion: {exc!r}") from exc
        result = parse_chat_completion(payload)
        if request.json_schema is None:
            return result
        try:
            parsed = json.loads(result.text)
        except ValueError:
            return result  # the gateway reports it and repairs
        normalized = normalize_arguments(parsed, request.json_schema)
        text = json.dumps(normalized, ensure_ascii=False)
        return LLMResponse(text=text, usage=result.usage, stop_reason=result.stop_reason)

    async def aclose(self) -> None:
        await self._client.aclose()
