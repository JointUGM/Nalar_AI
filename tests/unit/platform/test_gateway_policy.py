import asyncio

import pytest
from pydantic import BaseModel

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.gateway import CallPolicy, LLMGateway
from nalar_ai.platform.llm.ports import LLMRequest, LLMResponse, LLMUsage, TransientLLMError
from nalar_ai.platform.prompts.registry import PromptRegistry, parse_prompt
from nalar_ai.shared.enums import CallStatus, ModelTier
from nalar_ai.shared.errors import (
    ConfigurationError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import UsageLedger

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
=== task ===
Say {{word}}.
"""

MODELS = {
    ModelTier.FAST: "claude-haiku-4-5",
    ModelTier.QUALITY: "claude-sonnet-5",
    ModelTier.JUDGE: "claude-opus-5",
}

LIVE = CallPolicy(timeout_s=0.05, max_attempts=1, max_repairs=0, lane="live")


class Echo(BaseModel):
    word: str


class GatedLLM:
    """Replies at once, except to "Say wait." which blocks until `release` is set."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.blocks[-1].text == "Say wait.":
            await self.release.wait()
        return LLMResponse(text='{"word": "halo"}', usage=LLMUsage(input_tokens=10))


def _gateway(port: object, *, concurrency: int = 4) -> LLMGateway:
    return LLMGateway(
        port=port,  # type: ignore[arg-type]
        prompts=PromptRegistry([parse_prompt(PROMPT)]),
        models=MODELS,
        concurrency=concurrency,
    )


async def _generate(
    gateway: LLMGateway, ledger: UsageLedger, word: str, policy: CallPolicy
) -> Echo:
    return await gateway.generate(
        prompt_id="t.echo",
        variables={"word": word},
        output_model=Echo,
        ledger=ledger,
        policy=policy,
    )


async def test_a_slow_call_hits_its_deadline_and_is_recorded() -> None:
    port = GatedLLM()  # "wait" never gets released
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(UpstreamUnavailableError):
        await _generate(_gateway(port), ledger, "wait", LIVE)
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    assert record.error_message == "deadline exceeded after 0.05s"


async def test_live_policy_makes_no_repair_call() -> None:
    llm = ScriptedLLM(["not json", '{"word": "halo"}'])
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(OutputValidationError, match="no repair allowed"):
        await _generate(_gateway(llm), ledger, "halo", LIVE)
    assert len(llm.requests) == 1


async def test_policy_attempts_override_the_gateway_default() -> None:
    llm = ScriptedLLM([TransientLLMError("429"), '{"word": "halo"}'])
    ledger = UsageLedger("r", 1.0)
    with pytest.raises(UpstreamUnavailableError):
        await _generate(_gateway(llm), ledger, "halo", LIVE)
    assert len(ledger.records) == 1


async def test_live_calls_never_queue_behind_batch_calls() -> None:
    port = GatedLLM()
    gateway = _gateway(port, concurrency=1)
    ledger = UsageLedger("r", 1.0)
    batch = asyncio.create_task(_generate(gateway, ledger, "wait", CallPolicy()))
    await asyncio.sleep(0)  # the batch call now holds the only batch slot
    live = CallPolicy(timeout_s=1.0, max_attempts=1, max_repairs=0, lane="live")
    assert await _generate(gateway, ledger, "go", live) == Echo(word="halo")
    port.release.set()
    assert await batch == Echo(word="halo")


def test_effort_on_a_model_that_rejects_it_fails_at_startup() -> None:
    with pytest.raises(ConfigurationError, match="rejects effort"):
        LLMGateway(
            port=ScriptedLLM(),
            prompts=PromptRegistry([parse_prompt(PROMPT)]),
            models={**MODELS, ModelTier.QUALITY: "claude-haiku-4-5"},
        )
