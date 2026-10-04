from pathlib import Path

import pytest

from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import content_problems
from nalar_ai.subsystems.s5_insight_synthesizer.infrastructure.config_loader import (
    DEFAULT_CONFIG_PATH,
    load_synthesizer_config,
)


def test_the_shipped_config_loads() -> None:
    config = load_synthesizer_config()
    assert config.version == 2
    assert "satu" in config.numbers.number_words
    assert "sebagian besar" in config.numbers.count_claims
    assert config.insight.max_clusters == 6
    assert (config.summary.min_words, config.summary.max_words) == (60, 150)
    assert config.summary.banned_term_exceptions == (("salin", ("saling",)),)
    assert "menyalin" in config.summary.banned_terms


def test_a_missing_exception_config_keeps_stem_matching(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "  banned_term_exceptions:\n    salin:\n      - saling\n", ""
    )
    path = tmp_path / "synthesizer.yaml"
    path.write_text(text, encoding="utf-8")
    rules = load_synthesizer_config(path).summary
    assert rules.banned_term_exceptions == ()
    assert content_problems("Gaya dan gerak saling berkaitan.", rules) == [
        "do not mention scores, cheating or copying (found 'salin')"
    ]


def test_a_template_that_fails_the_guard_is_rejected(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
        "Ananda telah menyelesaikan", "Nilai Ananda telah menyelesaikan"
    )
    path = tmp_path / "synthesizer.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="template fails the guard"):
        load_synthesizer_config(path)
