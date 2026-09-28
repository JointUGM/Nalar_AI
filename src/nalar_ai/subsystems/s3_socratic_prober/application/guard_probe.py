"""Step 7: the leak guard around the adapted question (AI-9, design doc §11). Fails closed."""

from dataclasses import dataclass

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.shared.enums import GuardResult
from nalar_ai.shared.errors import (
    ModelCallRejectedError,
    OutputValidationError,
    UpstreamUnavailableError,
)
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import (
    GuardLexicon,
    GuardThresholds,
    check_similarity,
    check_text,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView

EMBEDDING_TAG = "guard"  # recorded as prompt_version "embed.guard"


@dataclass(frozen=True, slots=True)
class GuardOutcome:
    result: GuardResult
    degraded: bool = False  # the similarity checks could not run, so the guard failed closed

    @property
    def passed(self) -> bool:
        return self.result is GuardResult.PASSED


class GuardProbeUseCase:
    def __init__(
        self,
        *,
        embeddings: EmbeddingService,
        lexicon: GuardLexicon,
        thresholds: GuardThresholds,
    ) -> None:
        self._embeddings = embeddings
        self._lexicon = lexicon
        self._thresholds = thresholds

    async def execute(
        self,
        *,
        adapted: str,
        approved: str,
        pack: ContextPack,
        view: SessionView,
        ledger: UsageLedger,
        timeout_s: float | None,
    ) -> GuardOutcome:
        blocked = check_text(
            adapted, approved, view.student_answers(), pack.answer_terms, self._lexicon
        )
        if blocked is not None:
            return GuardOutcome(blocked)
        if timeout_s is None:
            return GuardOutcome(GuardResult.NOT_RUN, degraded=True)
        try:
            adapted_vec, approved_vec, reference_vec = await self._embeddings.embed(
                [adapted, approved, pack.reference_reasoning],
                tag=EMBEDDING_TAG,
                ledger=ledger,
                timeout_s=timeout_s,
                max_attempts=1,
            )
        except (UpstreamUnavailableError, ModelCallRejectedError, OutputValidationError):
            return GuardOutcome(GuardResult.NOT_RUN, degraded=True)
        return GuardOutcome(
            check_similarity(adapted_vec, approved_vec, reference_vec, self._thresholds)
        )
