"""S4 composition: the subsystem's long-lived collaborators, built once from settings."""

from dataclasses import dataclass
from pathlib import Path

from nalar_ai.settings import Settings
from nalar_ai.subsystems.s4_session_evaluator.application.policy import EvaluatorPolicy
from nalar_ai.subsystems.s4_session_evaluator.domain.config import EvaluatorConfig
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import VerificationLimits
from nalar_ai.subsystems.s4_session_evaluator.infrastructure.config_loader import (
    load_evaluator_config,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True, slots=True)
class S4Module:
    config: EvaluatorConfig
    policy: EvaluatorPolicy


def build_s4_module(settings: Settings) -> S4Module:
    return S4Module(
        config=load_evaluator_config(),
        policy=EvaluatorPolicy(
            score_timeout_s=settings.s4_score_timeout_seconds,
            reflect_timeout_s=settings.s4_reflect_timeout_seconds,
            max_attempts=settings.s4_max_attempts,
            max_answer_chars=settings.s4_max_answer_chars,
            limits=VerificationLimits(max_quotes_per_score=settings.s4_max_quotes_per_score),
        ),
    )
