"""Cache warm-up when a run starts (design doc §9): one call per S3 prompt, so the first of
32 students does not pay for writing the shared prefix. Never fails the run."""

import asyncio

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.enums import PlannerMode
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.choose_probe import (
    ChoiceFailure,
    ChooseProbeUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.application.classify_answer import (
    ClassifyAnswerUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.application.policy import ProberPolicy
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import plan_move
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack, validate_pack
from nalar_ai.subsystems.s3_socratic_prober.domain.session import (
    HistoryTurn,
    SessionView,
    TurnKind,
)

WARM_TIMEOUT_S = 10.0  # nobody waits on the warm-up, but it must not hang the request
WARM_ANSWER = "(pemanasan)"


class WarmCacheUseCase:
    def __init__(self, *, llm: LLMGateway, policy: ProberPolicy) -> None:
        self._classify = ClassifyAnswerUseCase(
            llm=llm,
            transcript_turns=policy.transcript_turns,
            max_answer_chars=policy.max_answer_chars,
        )
        self._choose = ChooseProbeUseCase(
            llm=llm,
            transcript_turns=policy.transcript_turns,
            max_answer_chars=policy.max_answer_chars,
        )

    async def execute(self, pack: ContextPack, ledger: UsageLedger) -> tuple[str, ...]:
        """Returns warnings, one per prompt that could not be warmed."""
        problems = validate_pack(pack)
        if problems:
            raise InvalidInputError("the context pack is invalid", details={"problems": problems})
        view = SessionView((HistoryTurn(0, TurnKind.ANCHOR, pack.anchor_problem, WARM_ANSWER),))
        plan = plan_move(Classification.fallback(), view, pack, mode=PlannerMode.HYBRID)
        classified, chosen = await asyncio.gather(
            self._classify.execute(pack, view, ledger, timeout_s=WARM_TIMEOUT_S),
            self._choose.execute(
                pack.writer_view(),
                view,
                Classification.fallback(),
                plan,
                ledger,
                timeout_s=WARM_TIMEOUT_S,
            ),
        )
        warnings: list[str] = []
        if classified.source is ClassificationSource.FALLBACK:
            warnings.append(f"warm_failed:{ClassifyAnswerUseCase.PROMPT_ID}")
        if isinstance(chosen, ChoiceFailure):
            warnings.append(f"warm_failed:{ChooseProbeUseCase.PROMPT_ID}")
        return tuple(warnings)
