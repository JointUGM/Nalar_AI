"""Pedagogy-owned content for the evaluator (evaluator.yaml)."""

from dataclasses import dataclass

from nalar_ai.subsystems.s4_session_evaluator.domain.reflection import (
    ReflectionLexicon,
    ReflectionParts,
)


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    version: int
    lexicon: ReflectionLexicon
    template: ReflectionParts  # shown when the model's reflection fails the guard twice
