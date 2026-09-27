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
