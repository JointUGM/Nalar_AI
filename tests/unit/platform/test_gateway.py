import logging
import uuid
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from nalar_ai.platform.llm.anthropic_adapter import AnthropicMessagesAdapter
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.llm.openai_chat_adapter import OpenAIChatAdapter
from nalar_ai.platform.llm.ports import (
    LLMPort,
    LLMResponse,
    LLMUsage,
    PermanentLLMError,
    TransientLLMError,
)
from nalar_ai.platform.llm.pricing import UnknownModelPriceError, llm_cost_usd
from nalar_ai.platform.prompts.registry import PromptRegistry, parse_prompt
from nalar_ai.shared.enums import CallStatus, ModelTier, RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import (
    BudgetExceededError,
    ModelCallRejectedError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger

PROMPT = """---
id: t.echo
version: 1
purpose: kb_extract
tier: quality
max_tokens: 1000
effort: medium
---
=== system ===
You echo words.
=== context ===
Material: {{material}}
=== task ===
Say {{word}}.
"""

MODELS = {
    ModelTier.FAST: "claude-haiku-4-5",
    ModelTier.QUALITY: "claude-sonnet-5",
    ModelTier.JUDGE: "claude-opus-5",
}


class Echo(BaseModel):
    word: str


def _gateway(llm: LLMPort, *, native: bool = True) -> tuple[LLMGateway, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    gateway = LLMGateway(
        port=llm,
        prompts=PromptRegistry([parse_prompt(PROMPT)]),
        models=MODELS,
        native_structured_output=native,
        sleep=fake_sleep,
    )
    return gateway, sleeps


async def _generate(gateway: LLMGateway, ledger: UsageLedger, **kwargs: object) -> Echo:
    return await gateway.generate(
        prompt_id="t.echo",
        variables={"material": "gaya", "word": "halo"},
        output_model=Echo,
        ledger=ledger,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_valid_first_answer_is_parsed_and_recorded() -> None:
    llm = ScriptedLLM(['{"word": "halo"}'])
    gateway, _ = _gateway(llm)
    ledger = UsageLedger("req-1", 1.0)
    ref = RetrievalRef(RetrievalSource.MATERIAL_CHUNK, uuid.uuid4(), RetrievalPath.LINK, 1)
    result = await _generate(gateway, ledger, retrieval=(ref,))
    assert result == Echo(word="halo")
    (record,) = ledger.records
    assert record.status is CallStatus.SUCCESS
    assert record.prompt_version == "t.echo@v1"
    assert record.model == "claude-sonnet-5"
    assert record.request_id == "req-1"
    assert record.retrieval == (ref,)
    assert record.cost_usd > 0
    request = llm.requests[0]
    assert request.system == "You echo words."
    assert request.blocks[0].text == "Material: gaya" and request.blocks[0].cache is True
    assert request.blocks[1].text == "Say halo." and request.blocks[1].cache is False
    assert request.json_schema is not None and request.effort == "medium"


async def test_code_fences_are_tolerated() -> None:
    gateway, _ = _gateway(ScriptedLLM(['```json\n{"word": "halo"}\n```']))
    assert await _generate(gateway, UsageLedger("r", 1.0)) == Echo(word="halo")


async def test_invalid_output_gets_exactly_one_repair() -> None:
    llm = ScriptedLLM(["not json", '{"word": "halo"}'])
    gateway, _ = _gateway(llm)
    ledger = UsageLedger("r", 1.0)
    assert await _generate(gateway, ledger) == Echo(word="halo")
    assert len(ledger.records) == 2
    repair_block = llm.requests[1].blocks[-1].text
    assert "could not be accepted" in repair_block
    assert "<previous_answer>" in repair_block


async def test_semantic_errors_fail_after_one_repair() -> None:
    llm = ScriptedLLM(['{"word": "x"}', '{"word": "y"}'])
    gateway, _ = _gateway(llm)
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(OutputValidationError) as info:
        await _generate(gateway, ledger, semantic_check=lambda out: ["word must be halo"])
    assert info.value.details["errors"] == ["word must be halo"]
    assert len(ledger.records) == 2
    assert "word must be halo" in llm.requests[1].blocks[-1].text


async def test_a_final_invalid_output_logs_which_checks_failed_without_their_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The 2026-09-30 502s were undiagnosable from the logs (P13). Messages may echo student
    # words, so only the path before the first colon is logged.
    llm = ScriptedLLM(['{"word": "x"}', '{"word": "y"}'])
    gateway, _ = _gateway(llm)
    errors = ["concepts[c2].initial.misconception: 'gayanya habis' is not listed", "no colon"]
    with (
        caplog.at_level(logging.WARNING, logger="nalar_ai.llm"),
        pytest.raises(OutputValidationError),
    ):
        await _generate(gateway, UsageLedger("req-9", 1.0), semantic_check=lambda out: errors)
    (line,) = [r.getMessage() for r in caplog.records if r.name == "nalar_ai.llm"]
    assert "t.echo@v1" in line and "req-9" in line
    assert "concepts[c2].initial.misconception" in line and "no colon" in line
    assert "gayanya" not in line


async def test_truncated_output_counts_as_invalid() -> None:
    truncated = LLMResponse(
        text='{"word": "ha', usage=LLMUsage(output_tokens=1000), stop_reason="max_tokens"
    )
    llm = ScriptedLLM([truncated, '{"word": "halo"}'])
    gateway, _ = _gateway(llm)
    assert await _generate(gateway, UsageLedger("r", 1.0)) == Echo(word="halo")
    assert "truncated" in llm.requests[1].blocks[-1].text


async def test_transient_errors_are_retried_with_backoff_and_recorded() -> None:
    llm = ScriptedLLM([TransientLLMError("429"), '{"word": "halo"}'])
    gateway, sleeps = _gateway(llm)
    ledger = UsageLedger("r", 1.0)
    assert await _generate(gateway, ledger) == Echo(word="halo")
    assert [r.status for r in ledger.records] == [CallStatus.ERROR, CallStatus.SUCCESS]
    assert ledger.records[0].error_message == "429"
    assert sleeps == [0.5]


async def test_exhausted_retries_raise_upstream_unavailable() -> None:
    llm = ScriptedLLM([TransientLLMError("503")] * 3)
    gateway, sleeps = _gateway(llm)
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(UpstreamUnavailableError):
        await _generate(gateway, ledger)
    assert len(ledger.records) == 3
    assert sleeps == [0.5, 1.0]


async def test_permanent_errors_are_not_retried() -> None:
    llm = ScriptedLLM([PermanentLLMError("400 bad request")])
    gateway, _ = _gateway(llm)
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(ModelCallRejectedError):
        await _generate(gateway, ledger)
    assert len(ledger.records) == 1


@pytest.mark.parametrize("api", ["chat", "messages"])
async def test_billed_refusal_is_rejected_and_usage_is_recorded(api: str) -> None:
    usage = LLMUsage(input_tokens=10, output_tokens=5, cache_read_tokens=3, cache_write_tokens=2)
    calls: list[object] = []
    if api == "chat":

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "refusal",
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "output",
                                            "arguments": '{"word": "halo"}',
                                        }
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 15,
                        "completion_tokens": 5,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                    },
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        adapter: LLMPort = OpenAIChatAdapter(
            client, base_url="https://sumopod.test/v1", api_key="k"
        )
    else:

        async def create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text='{"word": "halo"}')],
                stop_reason="refusal",
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    cache_read_input_tokens=3,
                    cache_creation_input_tokens=2,
                ),
            )

        adapter = AnthropicMessagesAdapter(SimpleNamespace(messages=SimpleNamespace(create=create)))

    gateway, sleeps = _gateway(adapter)
    ledger = UsageLedger("refusal", 1.0)
    try:
        with pytest.raises(ModelCallRejectedError, match="refused"):
            await _generate(gateway, ledger)
    finally:
        if api == "chat":
            await client.aclose()
    assert len(calls) == 1
    assert sleeps == []
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    assert record.error_message == "the model refused the request"
    assert (
        record.input_tokens,
        record.output_tokens,
        record.cache_read_tokens,
        record.cache_write_tokens,
    ) == (10, 5, 3, 2)
    assert record.cost_usd == round(llm_cost_usd("claude-sonnet-5", usage), 6)


async def test_budget_is_checked_before_calling() -> None:
    llm = ScriptedLLM(['{"word": "halo"}'])
    gateway, _ = _gateway(llm)
    with pytest.raises(BudgetExceededError):
        await _generate(gateway, UsageLedger("r", 0.0))
    assert llm.requests == []


async def test_prompt_mode_embeds_the_schema_in_the_task() -> None:
    llm = ScriptedLLM(['{"word": "halo"}'])
    gateway, _ = _gateway(llm, native=False)
    await _generate(gateway, UsageLedger("r", 1.0))
    assert llm.requests[0].json_schema is None
    assert "JSON Schema" in llm.requests[0].blocks[-1].text


def test_models_without_a_price_are_rejected_at_startup() -> None:
    with pytest.raises(UnknownModelPriceError):
        LLMGateway(
            port=ScriptedLLM(),
            prompts=PromptRegistry([]),
            models={**MODELS, ModelTier.FAST: "mystery-model"},
        )


async def test_unexpected_port_errors_are_recorded_and_rejected() -> None:
    llm = ScriptedLLM([RuntimeError("odd provider payload")])
    gateway, _ = _gateway(llm)
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(ModelCallRejectedError):
        await _generate(gateway, ledger)
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    assert "odd provider payload" in (record.error_message or "")
