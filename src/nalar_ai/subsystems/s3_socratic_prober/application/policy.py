"""Runtime limits for one turn (design doc §13), built from settings."""

from dataclasses import dataclass

from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardThresholds


@dataclass(frozen=True, slots=True)
class ProberPolicy:
    turn_budget_s: float
    classify_timeout_s: float
    choose_timeout_s: float
    embed_timeout_s: float
    min_step_s: float
    min_probes: int
    transcript_turns: int
    max_answer_chars: int
    guard: GuardThresholds
