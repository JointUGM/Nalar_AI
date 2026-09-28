"""The turn budget (design doc §13): each step gets min(its timeout, time left in the turn)."""

import time
from collections.abc import Callable


class Deadline:
    def __init__(
        self,
        budget_s: float,
        *,
        min_step_s: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock
        self._ends_at = clock() + budget_s
        self._min_step_s = min_step_s

    def step(self, limit_s: float) -> float | None:
        """Seconds this step may take, or None when too little time is left to try it."""
        allowed = min(limit_s, self._ends_at - self._clock())
        return allowed if allowed >= self._min_step_s else None
