from pathlib import Path

import pytest

from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s5_insight_synthesizer.infrastructure.config_loader import (
    DEFAULT_CONFIG_PATH,
    load_synthesizer_config,
)


def test_the_shipped_config_loads() -> None:
    config = load_synthesizer_config()
    assert config.version == 1
    assert "satu" in config.numbers.number_words
    assert "sebagian besar" in config.numbers.count_claims
    assert config.insight.max_clusters == 6
    assert (config.summary.min_words, config.summary.max_words) == (60, 150)


def test_a_template_that_fails_the_guard_is_rejected(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "Ananda telah menyelesaikan", "Nilai Ananda telah menyelesaikan"
    )
    path = tmp_path / "synthesizer.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="template fails the guard"):
        load_synthesizer_config(path)
