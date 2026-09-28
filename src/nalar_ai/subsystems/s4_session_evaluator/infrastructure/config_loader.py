from pathlib import Path
from typing import Any

import yaml

from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s4_session_evaluator.domain.config import EvaluatorConfig
from nalar_ai.subsystems.s4_session_evaluator.domain.reflection import (
    ReflectionLexicon,
    ReflectionParts,
    check_reflection,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "evaluator.yaml"


def load_evaluator_config(path: Path = DEFAULT_CONFIG_PATH) -> EvaluatorConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    reflection = raw["reflection"]
    lexicon = ReflectionLexicon.build(
        verdict_terms=reflection["verdict_terms"],
        banned_terms=reflection["banned_terms"],
        max_words=int(reflection["max_words"]),
    )
    template = ReflectionParts(**{key: str(value) for key, value in reflection["template"].items()})
    problems = check_reflection(template, (), (), lexicon)
    if problems:
        raise ConfigurationError(f"evaluator.yaml template fails the guard: {'; '.join(problems)}")
    return EvaluatorConfig(version=int(raw["version"]), lexicon=lexicon, template=template)
