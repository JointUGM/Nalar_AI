"""Pedagogy-owned content for the synthesizer (synthesizer.yaml)."""

from dataclasses import dataclass

from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import InsightLimits
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import NumberLexicon
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import (
    SummaryRules,
    SummaryTemplate,
)


@dataclass(frozen=True, slots=True)
class SynthesizerConfig:
    version: int
    numbers: NumberLexicon
    insight: InsightLimits
    summary: SummaryRules
    template: SummaryTemplate
