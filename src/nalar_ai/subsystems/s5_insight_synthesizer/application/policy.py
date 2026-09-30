"""Runtime limits for S5 calls (design doc §9), built from settings."""

from dataclasses import dataclass

from nalar_ai.platform.llm.gateway import CallPolicy


@dataclass(frozen=True, slots=True)
class SynthesizerPolicy:
    insight_timeout_s: float
    summary_timeout_s: float
    max_attempts: int

    def insight_call(self) -> CallPolicy:
        """No repair: a blocked draft gets a fresh call with feedback, not a patch."""
        return CallPolicy(
            timeout_s=self.insight_timeout_s,
            max_attempts=self.max_attempts,
            max_repairs=0,
            lane="scoring",
        )

    def summary_call(self) -> CallPolicy:
        return CallPolicy(
            timeout_s=self.summary_timeout_s,
            max_attempts=self.max_attempts,
            max_repairs=0,
            lane="scoring",
        )
