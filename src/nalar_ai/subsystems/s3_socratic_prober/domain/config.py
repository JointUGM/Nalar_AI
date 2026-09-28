"""Pedagogy-owned content for the prober: lexicons and teacher-facing texts (prober.yaml)."""

from collections.abc import Mapping
from dataclasses import dataclass

from nalar_ai.shared.enums import MoveSource, ProbeStrategy
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardLexicon
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.prefilter import PrefilterLexicon


@dataclass(frozen=True, slots=True)
class ProberTexts:
    """Indonesian texts shown to people. Templates use {move} and {answer}."""

    safety_message: str
    default_template: str
    coverage_template: str
    fallback_error_template: str
    fallback_invalid_template: str
    move_labels: Mapping[ProbeStrategy, str]
    answer_labels: Mapping[AnswerType, str]

    def reason(
        self,
        *,
        move: ProbeStrategy,
        answer_type: AnswerType,
        move_source: MoveSource,
        coverage_forced: bool,
    ) -> str:
        """The plain-language "why this question" for the teacher's report (session_turns.move_reason)."""
        if move_source is MoveSource.FALLBACK_ERROR:
            template = self.fallback_error_template
        elif move_source is MoveSource.FALLBACK_INVALID:
            template = self.fallback_invalid_template
        elif coverage_forced:
            template = self.coverage_template
        else:
            template = self.default_template
        return template.format(move=self.move_labels[move], answer=self.answer_labels[answer_type])


@dataclass(frozen=True, slots=True)
class ProberConfig:
    version: int
    prefilter: PrefilterLexicon
    guard: GuardLexicon
    texts: ProberTexts
