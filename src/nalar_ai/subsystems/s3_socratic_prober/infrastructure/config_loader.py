from pathlib import Path
from typing import Any

import yaml

from nalar_ai.shared.enums import ProbeStrategy
from nalar_ai.shared.errors import ConfigurationError
from nalar_ai.subsystems.s3_socratic_prober.domain.config import ProberConfig, ProberTexts
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardLexicon
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.prefilter import PrefilterLexicon

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "prober.yaml"


def load_prober_config(path: Path = DEFAULT_CONFIG_PATH) -> ProberConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    prefilter, guard, texts = raw["prefilter"], raw["guard"], raw["texts"]
    moves = {ProbeStrategy(key): str(value) for key, value in texts["moves"].items()}
    answers = {AnswerType(key): str(value) for key, value in texts["answers"].items()}
    missing = [m.value for m in ProbeStrategy if m not in moves] + [
        a.value for a in AnswerType if a not in answers
    ]
    if missing:
        raise ConfigurationError(f"prober.yaml has no label for: {', '.join(missing)}")
    templates = texts["templates"]
    return ProberConfig(
        version=int(raw["version"]),
        prefilter=PrefilterLexicon.build(
            safety_phrases=prefilter["safety_phrases"],
            evasive_answers=prefilter["evasive_answers"],
            manipulation_phrases=prefilter["manipulation_phrases"],
            max_evasive_chars=int(prefilter["max_evasive_chars"]),
            max_manipulation_words=int(prefilter["max_manipulation_words"]),
        ),
        guard=GuardLexicon.build(
            verdict_terms=guard["verdict_terms"],
            max_chars=int(guard["max_chars"]),
            max_sentences=int(guard["max_sentences"]),
        ),
        texts=ProberTexts(
            safety_message=str(texts["safety_message"]).strip(),
            default_template=str(templates["default"]),
            coverage_template=str(templates["coverage"]),
            fallback_error_template=str(templates["fallback_error"]),
            fallback_invalid_template=str(templates["fallback_invalid"]),
            move_labels=moves,
            answer_labels=answers,
        ),
    )
