from pathlib import Path
from typing import Any

import yaml

from nalar_ai.shared.enums import ConceptOutcome
from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s5_insight_synthesizer.domain.config import SynthesizerConfig
from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import InsightLimits
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import NumberLexicon
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    ParentConcept,
    SummaryRules,
    SummaryTemplate,
    content_problems,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "synthesizer.yaml"

# Renders every template sentence once, so the loader can guard the whole text.
_SAMPLE = (
    ParentConcept("gaya", ConceptOutcome.MASTERED, None, False),
    ParentConcept("gerak", ConceptOutcome.DEVELOPING, None, False),
)


def load_synthesizer_config(path: Path = DEFAULT_CONFIG_PATH) -> SynthesizerConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    numbers = NumberLexicon.build(
        number_words=raw["numbers"]["number_words"], count_claims=raw["numbers"]["count_claims"]
    )
    insight = InsightLimits(**{key: int(value) for key, value in raw["insight"].items()})
    summary = raw["summary"]
    rules = SummaryRules.build(
        min_words=int(summary["min_words"]),
        max_words=int(summary["max_words"]),
        banned_terms=summary["banned_terms"],
        comparison_phrases=summary["comparison_phrases"],
    )
    template = SummaryTemplate(**{k: str(v) for k, v in summary["template"].items()})
    problems = content_problems(template.render(_SAMPLE), rules)
    if problems:
        raise ConfigurationError(
            f"synthesizer.yaml template fails the guard: {'; '.join(problems)}"
        )
    return SynthesizerConfig(
        version=int(raw["version"]),
        numbers=numbers,
        insight=insight,
        summary=rules,
        template=template,
    )
