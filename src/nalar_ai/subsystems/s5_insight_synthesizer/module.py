"""S5 composition: the subsystem's long-lived collaborators, built once from settings."""

from dataclasses import dataclass
from pathlib import Path

from nalar_ai.settings import Settings
from nalar_ai.subsystems.s5_insight_synthesizer.application.policy import SynthesizerPolicy
from nalar_ai.subsystems.s5_insight_synthesizer.domain.config import SynthesizerConfig
from nalar_ai.subsystems.s5_insight_synthesizer.infrastructure.config_loader import (
    load_synthesizer_config,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True, slots=True)
class S5Module:
    config: SynthesizerConfig
    policy: SynthesizerPolicy


def build_s5_module(settings: Settings) -> S5Module:
    return S5Module(
        config=load_synthesizer_config(),
        policy=SynthesizerPolicy(
            insight_timeout_s=settings.s5_insight_timeout_seconds,
            summary_timeout_s=settings.s5_summary_timeout_seconds,
            max_attempts=settings.s5_max_attempts,
        ),
    )
