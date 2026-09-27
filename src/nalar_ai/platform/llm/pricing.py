"""SumoPod list prices (PRD §7.5), US$ per million tokens."""

from collections.abc import Mapping
from dataclasses import dataclass

from nalar_ai.platform.llm.ports import LLMUsage
from nalar_ai.shared.errors import ConfigurationError

_PER = 1_000_000


class UnknownModelPriceError(ConfigurationError):
    code = "unknown_model_price"


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float
    # SumoPod lists no cache-write price; Anthropic charges 1.25x input for 5-minute writes.
    cache_write_multiplier: float = 1.25


LLM_PRICES: Mapping[str, ModelPrice] = {
    "claude-haiku-4-5": ModelPrice(1.0, 0.10, 5.0),
    "claude-sonnet-5": ModelPrice(2.0, 0.20, 10.0),
    "claude-opus-5": ModelPrice(5.0, 0.50, 25.0),
}

EMBEDDING_PRICES_PER_MTOK: Mapping[str, float] = {"text-embedding-3-small": 0.02}


def llm_cost_usd(model: str, usage: LLMUsage) -> float:
    price = LLM_PRICES.get(model)
    if price is None:
        raise UnknownModelPriceError(f"no price configured for model {model!r}")
    return (
        usage.input_tokens * price.input_per_mtok
        + usage.cache_read_tokens * price.cached_input_per_mtok
        + usage.cache_write_tokens * price.input_per_mtok * price.cache_write_multiplier
        + usage.output_tokens * price.output_per_mtok
    ) / _PER


def embedding_cost_usd(model: str, tokens: int) -> float:
    price = EMBEDDING_PRICES_PER_MTOK.get(model)
    if price is None:
        raise UnknownModelPriceError(f"no price configured for embedding model {model!r}")
    return tokens * price / _PER
