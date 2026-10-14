from pathlib import Path
from typing import Any

import yaml

from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s5_insight_synthesizer.domain.config import SynthesizerConfig
from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import InsightLimits
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import NumberLexicon

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "synthesizer.yaml"


def load_synthesizer_config(path: Path = DEFAULT_CONFIG_PATH) -> SynthesizerConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    numbers = NumberLexicon.build(
        number_words=raw["numbers"]["number_words"], count_claims=raw["numbers"]["count_claims"]
    )
    try:
        insight = InsightLimits(**raw["insight"])
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid S5 insight limits: {error}") from error
    return SynthesizerConfig(
        version=int(raw["version"]),
        numbers=numbers,
        insight=insight,
    )
