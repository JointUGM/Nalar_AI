"""The one gateway for every model call (AI-12)."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from nalar_ai.platform.llm.ports import (
    LLMPort,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    PermanentLLMError,
    TransientLLMError,
    UserBlock,
)
from nalar_ai.platform.llm.pricing import LLM_PRICES, UnknownModelPriceError, llm_cost_usd
from nalar_ai.platform.llm.schema import strict_json_schema
from nalar_ai.platform.prompts.registry import PromptRegistry, PromptTemplate
from nalar_ai.platform.retry import retry_transient
from nalar_ai.shared.enums import CallStatus, ModelTier
from nalar_ai.shared.errors import (
    ConfigurationError,
    ModelCallRejectedError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import InvocationRecord, RetrievalRef, UsageLedger
from nalar_ai.shared.text import fence_untrusted

_MAX_REPORTED_ERRORS = 20
_MAX_ECHOED_OUTPUT_CHARS = 20_000

# Models whose API rejects `output_config.effort` (Haiku 4.5 returns 400).
MODELS_WITHOUT_EFFORT = frozenset({"claude-haiku-4-5"})

Lane = Literal["batch", "live"]


@dataclass(frozen=True, slots=True)
class CallPolicy:
    """Per-call limits. The defaults suit batch work (S1); live turns (S3) pass a tight policy.

    timeout_s: per attempt, including the wait for a concurrency slot; None = no deadline.
    max_attempts: None = the gateway default. max_repairs: repair calls for invalid output.
    lane: live calls never queue behind batch calls.
    """

    timeout_s: float | None = None
    max_attempts: int | None = None
    max_repairs: int = 1
    lane: Lane = "batch"


BATCH_POLICY = CallPolicy()


class LLMGateway:
    def __init__(
        self,
        *,
        port: LLMPort,
        prompts: PromptRegistry,
        models: Mapping[ModelTier, str],
        native_structured_output: bool = True,
        max_attempts: int = 3,
        concurrency: int = 4,
        live_concurrency: int = 32,
        base_delay_s: float = 0.5,
        provider: str = "sumopod",
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        for tier, model in models.items():
            if model not in LLM_PRICES:
                raise UnknownModelPriceError(f"tier {tier.value}: no price for model {model!r}")
        for template in prompts.templates():
            mapped = models.get(template.tier)
            if template.effort is not None and mapped in MODELS_WITHOUT_EFFORT:
                raise ConfigurationError(
                    f"{template.version_tag} sets effort, but its tier {template.tier.value} "
                    f"maps to {mapped}, which rejects effort"
                )
        self._port = port
        self._prompts = prompts
        self._models = dict(models)
        self._native = native_structured_output
        self._max_attempts = max_attempts
        self._semaphores: dict[Lane, asyncio.Semaphore] = {
            "batch": asyncio.Semaphore(concurrency),
            "live": asyncio.Semaphore(live_concurrency),
        }
        self._base_delay_s = base_delay_s
        self._provider = provider
        self._clock = clock
        self._sleep = sleep

    async def generate[M: BaseModel](
        self,
        *,
        prompt_id: str,
        variables: Mapping[str, str],
        output_model: type[M],
        ledger: UsageLedger,
        retrieval: Sequence[RetrievalRef] = (),
        semantic_check: Callable[[M], Sequence[str]] | None = None,
        policy: CallPolicy = BATCH_POLICY,
    ) -> M:
        template = self._prompts.get(prompt_id)
        rendered = template.render(variables)
        schema = strict_json_schema(output_model)
        blocks: list[UserBlock] = []
        if rendered.context:
            blocks.append(UserBlock(rendered.context, cache=True))
        task = (
            rendered.task if self._native else f"{rendered.task}\n\n{_schema_instruction(schema)}"
        )
        blocks.append(UserBlock(task))
        request = LLMRequest(
            model=self._models[template.tier],
            system=rendered.system,
            blocks=tuple(blocks),
            max_tokens=template.max_tokens,
            json_schema=schema if self._native else None,
            effort=template.effort,
        )

        response = await self._call(request, template, ledger, retrieval, policy)
        parsed, errors = _parse(response, output_model, semantic_check)
        if parsed is not None:
            return parsed

        for _ in range(policy.max_repairs):
            repair = replace(
                request,
                blocks=(*request.blocks, UserBlock(_repair_instruction(response.text, errors))),
            )
            response = await self._call(repair, template, ledger, retrieval, policy)
            parsed, errors = _parse(response, output_model, semantic_check)
            if parsed is not None:
                return parsed
        raise OutputValidationError(
            f"{template.version_tag} returned invalid output{_after_repairs(policy.max_repairs)}",
            details={"errors": list(errors)[:_MAX_REPORTED_ERRORS]},
        )

    async def _call(
        self,
        request: LLMRequest,
        template: PromptTemplate,
        ledger: UsageLedger,
        retrieval: Sequence[RetrievalRef],
        policy: CallPolicy,
    ) -> LLMResponse:
        semaphore = self._semaphores[policy.lane]

        async def attempt() -> LLMResponse:
            ledger.ensure_budget()
            started = self._clock()
            try:
                async with asyncio.timeout(policy.timeout_s), semaphore:
                    response = await self._port.complete(request)
            except TimeoutError as exc:
                detail = f"deadline exceeded after {policy.timeout_s}s"
                ledger.add(
                    self._record(
                        template, request.model, LLMUsage(), started, ledger, retrieval, detail
                    )
                )
                raise TransientLLMError(detail) from exc
            except (TransientLLMError, PermanentLLMError) as exc:
                ledger.add(
                    self._record(
                        template, request.model, LLMUsage(), started, ledger, retrieval, str(exc)
                    )
                )
                raise
            except Exception as exc:  # an odd payload must not drop the attempt's record
                detail = f"unexpected provider error: {exc!r}"
                ledger.add(
                    self._record(
                        template, request.model, LLMUsage(), started, ledger, retrieval, detail
                    )
                )
                raise PermanentLLMError(detail) from exc
            ledger.add(
                self._record(template, request.model, response.usage, started, ledger, retrieval)
            )
            return response

        try:
            return await retry_transient(
                attempt,
                is_transient=lambda exc: isinstance(exc, TransientLLMError),
                max_attempts=policy.max_attempts or self._max_attempts,
                base_delay_s=self._base_delay_s,
                sleep=self._sleep,
            )
        except TransientLLMError as exc:
            raise UpstreamUnavailableError(f"model provider unavailable: {exc}") from exc
        except PermanentLLMError as exc:
            raise ModelCallRejectedError(f"model call rejected: {exc}") from exc

    def _record(
        self,
        template: PromptTemplate,
        model: str,
        usage: LLMUsage,
        started: float,
        ledger: UsageLedger,
        retrieval: Sequence[RetrievalRef],
        error: str | None = None,
    ) -> InvocationRecord:
        return InvocationRecord(
            purpose=template.purpose,
            model=model,
            prompt_version=template.version_tag,
            status=CallStatus.ERROR if error else CallStatus.SUCCESS,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            latency_ms=int((self._clock() - started) * 1000),
            cost_usd=round(llm_cost_usd(model, usage), 6),
            request_id=ledger.request_id,
            provider=self._provider,
            error_message=error,
            retrieval=tuple(retrieval),
        )


def _after_repairs(max_repairs: int) -> str:
    if max_repairs == 0:
        return " (no repair allowed)"
    if max_repairs == 1:
        return " after one repair"
    return f" after {max_repairs} repairs"


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _parse[M: BaseModel](
    response: LLMResponse,
    output_model: type[M],
    semantic_check: Callable[[M], Sequence[str]] | None,
) -> tuple[M | None, list[str]]:
    if response.stop_reason == "max_tokens":
        return None, ["the answer was truncated at max_tokens; return a shorter complete answer"]
    try:
        parsed = output_model.model_validate_json(_extract_json(response.text))
    except ValidationError as exc:
        return None, [
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors()
        ]
    errors = list(semantic_check(parsed)) if semantic_check is not None else []
    return (None, errors) if errors else (parsed, [])


def _repair_instruction(previous: str, errors: Sequence[str]) -> str:
    listed = "\n".join(f"- {error}" for error in errors[:_MAX_REPORTED_ERRORS])
    return (
        "Your previous answer could not be accepted.\n"
        f"Problems:\n{listed}\n\n"
        "Previous answer:\n"
        f"{fence_untrusted('previous_answer', previous[:_MAX_ECHOED_OUTPUT_CHARS])}\n\n"
        "Return the complete corrected answer as JSON only, following the same rules and schema."
    )


def _schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with one JSON object only (no prose, no code fences) that validates against "
        "this JSON Schema:\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
