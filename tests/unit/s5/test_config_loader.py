from pathlib import Path

import pytest

from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s5_insight_synthesizer.infrastructure.config_loader import (
    DEFAULT_CONFIG_PATH,
    load_synthesizer_config,
)


def test_the_shipped_config_loads() -> None:
    config = load_synthesizer_config()
    assert config.version == 4
    assert "satu" in config.numbers.number_words
    assert "sebagian besar" in config.numbers.count_claims
    assert config.insight.max_clusters == 6
    assert (
        config.insight.max_suggestions,
        config.insight.suggestion_max_words,
        config.insight.suggestion_max_chars,
    ) == (3, 60, 600)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_suggestions", "-1"),
        ("max_suggestions", "true"),
        ("suggestion_max_words", "0"),
        ("suggestion_max_words", "60.5"),
        ("suggestion_max_chars", "0"),
        ("suggestion_max_chars", '"600"'),
    ],
)
def test_invalid_suggestion_limits_reject_configuration(
    tmp_path: Path, field: str, value: str
) -> None:
    import yaml

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["insight"][field] = yaml.safe_load(value)
    path = tmp_path / "synthesizer.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid S5 insight limits"):
        load_synthesizer_config(path)
