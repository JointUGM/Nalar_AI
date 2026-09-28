"""POST /v1/s4/sessions/evaluate: validate, score, reflect (design doc §4)."""

import logging
from dataclasses import dataclass

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s4_session_evaluator.application.policy import EvaluatorPolicy
from nalar_ai.subsystems.s4_session_evaluator.application.score_session import (
    ScoreSessionUseCase,
)
from nalar_ai.subsystems.s4_session_evaluator.application.write_reflection import (
    ReflectionResult,
    WriteReflectionUseCase,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.config import EvaluatorConfig
from nalar_ai.subsystems.s4_session_evaluator.domain.session import EvalSession, validate_session
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import Evaluation

logger = logging.getLogger("nalar_ai.s4")


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    evaluation: Evaluation
    reflection: ReflectionResult
    warnings: tuple[str, ...]


class EvaluateSessionUseCase:
    def __init__(
        self, *, llm: LLMGateway, config: EvaluatorConfig, policy: EvaluatorPolicy
    ) -> None:
        self._score = ScoreSessionUseCase(llm=llm, policy=policy)
        self._reflect = WriteReflectionUseCase(llm=llm, config=config, policy=policy)
        self._config = config

    async def execute(self, session: EvalSession, ledger: UsageLedger) -> EvaluationOutcome:
        """Stateless: the backend guarantees each session is sent once (NFR-R2)."""
        problems = validate_session(session)
        if problems:
            raise InvalidInputError(
                "the session cannot be evaluated", details={"problems": problems}
            )
        scored = await self._score.execute(session, ledger)
        reflection = await self._reflect.execute(session, scored.evaluation, ledger)
        warnings: list[str] = []
        if scored.repaired:
            warnings.append("scoring_repaired")
        if reflection.source == "template":
            warnings.append("reflection_fallback")
        elif reflection.retried:
            warnings.append("reflection_retried")
        logger.info(
            "s4 evaluation request_id=%s turns=%d answered=%d levels=%s repaired=%s "
            "reflection=%s calls=%d cost_usd=%.6f lexicon_v=%d",
            ledger.request_id,
            len(session.turns),
            len(session.answered_turns),
            ",".join(str(score.level) for score in scored.evaluation.scores),
            scored.repaired,
            reflection.source,
            len(ledger.records),
            ledger.total_cost_usd,
            self._config.version,
        )
        return EvaluationOutcome(scored.evaluation, reflection, tuple(warnings))
