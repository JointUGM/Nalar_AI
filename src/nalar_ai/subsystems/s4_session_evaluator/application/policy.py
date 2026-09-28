"""Runtime limits for one evaluation (design doc §11), built from settings."""

from dataclasses import dataclass

from nalar_ai.platform.llm.gateway import CallPolicy
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import VerificationLimits


@dataclass(frozen=True, slots=True)
class EvaluatorPolicy:
    score_timeout_s: float
    reflect_timeout_s: float
    max_attempts: int
    max_answer_chars: int
    limits: VerificationLimits

    def score_call(self) -> CallPolicy:
        """One repair: an unverifiable answer gets exactly one chance to fix its quotes."""
        return CallPolicy(
            timeout_s=self.score_timeout_s,
            max_attempts=self.max_attempts,
            max_repairs=1,
            lane="scoring",
        )

    def reflect_call(self) -> CallPolicy:
        """No repair: a blocked reflection gets a fresh call, not a patch (design doc §9)."""
        return CallPolicy(
            timeout_s=self.reflect_timeout_s,
            max_attempts=self.max_attempts,
            max_repairs=0,
            lane="scoring",
        )
