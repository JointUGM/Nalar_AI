"""Day-one SumoPod verification (design doc §12.5). Costs a few US cents.

Needs a real .env (NALAR_AI_SUMOPOD_API_KEY and both base URLs).
Run: uv run python scripts/sumopod_probe.py
"""

import asyncio
import statistics
import time

from pydantic import BaseModel

from nalar_ai.platform.embeddings.openai_compat import OpenAICompatEmbedder
from nalar_ai.platform.llm.anthropic_adapter import AnthropicMessagesAdapter
from nalar_ai.platform.llm.ports import LLMRequest, UserBlock
from nalar_ai.platform.llm.schema import strict_json_schema
from nalar_ai.settings import get_settings

# About 2,000 tokens, above the 1,024-token minimum cacheable prefix of Sonnet 5.
_FILLER = (
    "Gaya gesek adalah gaya yang melawan arah gerak benda ketika dua permukaan bersentuhan. " * 80
)


class _Answer(BaseModel):
    answer: str


async def main() -> None:
    settings = get_settings()
    llm = AnthropicMessagesAdapter.from_settings(settings)
    embedder = OpenAICompatEmbedder.from_settings(settings)
    results: list[tuple[str, bool, str]] = []

    def ping(model: str) -> LLMRequest:
        return LLMRequest(
            model=model,
            system="Reply with the single word OK.",
            blocks=(UserBlock("Ping"),),
            max_tokens=20,
        )

    try:
        try:
            started = time.perf_counter()
            reply = await llm.complete(ping(settings.model_fast))
            elapsed = time.perf_counter() - started
            results.append(
                ("messages endpoint", "ok" in reply.text.lower(), f"{reply.text!r} {elapsed:.2f}s")
            )
        except Exception as exc:
            results.append(("messages endpoint", False, repr(exc)))

        try:
            cached = LLMRequest(
                model=settings.model_quality,
                system="Answer in one short sentence.",
                blocks=(UserBlock(_FILLER, cache=True), UserBlock("Apa itu gaya gesek?")),
                max_tokens=200,
            )
            await llm.complete(cached)
            second = await llm.complete(cached)
            usage = second.usage
            results.append(
                (
                    "prompt caching",
                    usage.cache_read_tokens > 0,
                    f"cache_read={usage.cache_read_tokens} cache_write={usage.cache_write_tokens}",
                )
            )
        except Exception as exc:
            results.append(("prompt caching", False, repr(exc)))

        try:
            structured = await llm.complete(
                LLMRequest(
                    model=settings.model_fast,
                    system="Answer the question.",
                    blocks=(UserBlock("Apa ibu kota Indonesia?"),),
                    max_tokens=200,
                    json_schema=strict_json_schema(_Answer),
                )
            )
            _Answer.model_validate_json(structured.text)
            results.append(("structured output", True, structured.text))
        except Exception as exc:
            results.append(("structured output", False, repr(exc)))

        try:
            timings: list[float] = []
            for _ in range(10):
                started = time.perf_counter()
                await llm.complete(ping(settings.model_fast))
                timings.append(time.perf_counter() - started)
            p50 = statistics.median(timings)
            p95 = statistics.quantiles(timings, n=20)[18]
            results.append(("haiku latency", p50 < 1.5, f"p50={p50:.2f}s p95={p95:.2f}s"))
        except Exception as exc:
            results.append(("haiku latency", False, repr(exc)))

        # S3's chooser call: Sonnet 5 + effort + structured output together (design doc s3 R-S3-2).
        try:
            timings = []
            for _ in range(5):
                started = time.perf_counter()
                reply = await llm.complete(
                    LLMRequest(
                        model=settings.model_quality,
                        system="Answer the question in Indonesian.",
                        blocks=(UserBlock(_FILLER, cache=True), UserBlock("Apa itu gaya gesek?")),
                        max_tokens=300,
                        json_schema=strict_json_schema(_Answer),
                        effort="low",
                    )
                )
                timings.append(time.perf_counter() - started)
                _Answer.model_validate_json(reply.text)
            p50 = statistics.median(timings)
            results.append(
                ("sonnet effort+json", p50 < 2.5, f"p50={p50:.2f}s max={max(timings):.2f}s")
            )
        except Exception as exc:
            results.append(("sonnet effort+json", False, repr(exc)))

        # A class of 32 answering at once (NFR-P2): no rate-limit errors, p95 still usable.
        try:

            async def timed_ping() -> float:
                started = time.perf_counter()
                await llm.complete(ping(settings.model_fast))
                return time.perf_counter() - started

            burst = await asyncio.gather(*(timed_ping() for _ in range(32)), return_exceptions=True)
            failures = [r for r in burst if isinstance(r, BaseException)]
            timings = [r for r in burst if isinstance(r, float)]
            p95 = statistics.quantiles(timings, n=20)[18] if len(timings) > 1 else float("nan")
            results.append(
                (
                    "burst of 32",
                    not failures and p95 < 3.0,
                    f"errors={len(failures)} p95={p95:.2f}s"
                    + (f" first_error={failures[0]!r}" if failures else ""),
                )
            )
        except Exception as exc:
            results.append(("burst of 32", False, repr(exc)))

        try:
            batch = await embedder.embed(
                ["gaya gesek"],
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
            dims = len(batch.vectors[0])
            results.append(("embeddings", dims == settings.embedding_dimensions, f"dims={dims}"))
        except Exception as exc:
            results.append(("embeddings", False, repr(exc)))
    finally:
        await llm.aclose()
        await embedder.aclose()

    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<20} {detail}")


if __name__ == "__main__":
    asyncio.run(main())
