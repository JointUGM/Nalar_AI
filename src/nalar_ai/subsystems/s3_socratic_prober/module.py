"""S3 composition: the subsystem's long-lived collaborators, built once from settings."""

from dataclasses import dataclass
from pathlib import Path

from nalar_ai.settings import Settings
from nalar_ai.subsystems.s3_socratic_prober.application.policy import ProberPolicy
from nalar_ai.subsystems.s3_socratic_prober.domain.config import ProberConfig
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardThresholds
from nalar_ai.subsystems.s3_socratic_prober.infrastructure.config_loader import (
    load_prober_config,
)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True, slots=True)
class S3Module:
    config: ProberConfig
    policy: ProberPolicy


def build_s3_module(settings: Settings) -> S3Module:
    return S3Module(
        config=load_prober_config(),
        policy=ProberPolicy(
            turn_budget_s=settings.s3_turn_budget_seconds,
            classify_timeout_s=settings.s3_classify_timeout_seconds,
            choose_timeout_s=settings.s3_choose_timeout_seconds,
            embed_timeout_s=settings.s3_embed_timeout_seconds,
            min_step_s=settings.s3_min_step_seconds,
            min_probes=settings.s3_min_probes,
            transcript_turns=settings.s3_transcript_turns,
            max_answer_chars=settings.s3_max_answer_chars,
            guard=GuardThresholds(
                max_reference_similarity=settings.s3_guard_max_reference_similarity,
                min_approved_similarity=settings.s3_guard_min_approved_similarity,
            ),
        ),
    )
