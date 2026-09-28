import pytest

from nalar_ai.platform.llm.ports import LLMUsage
from nalar_ai.platform.llm.pricing import UnknownModelPriceError, embedding_cost_usd, llm_cost_usd

MTOK = 1_000_000


def test_sonnet_prices_per_million_tokens() -> None:
    assert llm_cost_usd("claude-sonnet-5", LLMUsage(input_tokens=MTOK)) == pytest.approx(2.0)
    assert llm_cost_usd("claude-sonnet-5", LLMUsage(cache_read_tokens=MTOK)) == pytest.approx(0.2)
    assert llm_cost_usd("claude-sonnet-5", LLMUsage(cache_write_tokens=MTOK)) == pytest.approx(2.5)
    assert llm_cost_usd("claude-sonnet-5", LLMUsage(output_tokens=MTOK)) == pytest.approx(10.0)


def test_haiku_and_embedding_prices() -> None:
    usage = LLMUsage(input_tokens=1000, output_tokens=100)
    assert llm_cost_usd("claude-haiku-4-5", usage) == pytest.approx(0.0015)
    assert embedding_cost_usd("text-embedding-3-small", MTOK) == pytest.approx(0.02)


def test_unknown_model_has_no_silent_zero_price() -> None:
    with pytest.raises(UnknownModelPriceError):
        llm_cost_usd("gpt-unknown", LLMUsage(input_tokens=1))


@pytest.mark.parametrize(
    ("model", "input_price", "cached_price", "output_price"),
    [
        ("gpt-5.4-mini", 0.75, 0.075, 4.5),
        ("gpt-5.4", 2.5, 0.25, 15.0),
        ("qwen3.8-max", 1.0, 0.125, 3.0),
    ],
)
def test_benchmarked_non_claude_prices(
    model: str, input_price: float, cached_price: float, output_price: float
) -> None:
    # SumoPod list prices, 2026-09-28 (DECISIONS M3-M7). qwen3.8-max is at its 50%-off price.
    assert llm_cost_usd(model, LLMUsage(input_tokens=MTOK)) == pytest.approx(input_price)
    assert llm_cost_usd(model, LLMUsage(cache_read_tokens=MTOK)) == pytest.approx(cached_price)
    assert llm_cost_usd(model, LLMUsage(output_tokens=MTOK)) == pytest.approx(output_price)
    # These providers cache implicitly and list no write surcharge.
    assert llm_cost_usd(model, LLMUsage(cache_write_tokens=MTOK)) == pytest.approx(input_price)


def test_every_default_model_has_a_price() -> None:
    from nalar_ai.platform.llm.pricing import LLM_PRICES
    from nalar_ai.settings import Settings

    defaults = Settings.model_fields
    for tier in ("model_fast", "model_quality", "model_judge"):
        assert defaults[tier].default in LLM_PRICES


def test_default_tiers_follow_the_benchmark() -> None:
    from nalar_ai.settings import Settings

    defaults = Settings.model_fields
    assert defaults["model_fast"].default == "gpt-5.4-mini"  # M3, M4
    assert defaults["model_quality"].default == "claude-sonnet-5"  # M5: kept for the pilot
    assert defaults["model_judge"].default == "qwen3.8-max"  # M6
