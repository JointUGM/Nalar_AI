from collections.abc import Sequence
from dataclasses import replace

from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.embeddings.ports import EmbeddingBatch, TransientEmbeddingError
from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.shared.enums import CallStatus, GuardResult
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.guard_probe import (
    GuardOutcome,
    GuardProbeUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardThresholds
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack
from nalar_ai.subsystems.s3_socratic_prober.domain.session import SessionView
from nalar_ai.subsystems.s3_socratic_prober.infrastructure.config_loader import (
    load_prober_config,
)
from tests.support.s3 import T_LEMBAM, anchor, make_pack, question

APPROVED = question(T_LEMBAM, M.COUNTER_EXAMPLE).text
VIEW = SessionView((anchor("karena gayanya habis"),))
LEXICON = load_prober_config().guard


class FailingEmbedder:
    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        raise TransientEmbeddingError("503")


def _use_case(
    embedder: object | None = None, thresholds: GuardThresholds | None = None
) -> GuardProbeUseCase:
    return GuardProbeUseCase(
        embeddings=EmbeddingService(
            embedder or HashingEmbedder(),  # type: ignore[arg-type]
            model="text-embedding-3-small",
            dimensions=1536,
        ),
        lexicon=LEXICON,
        thresholds=thresholds or GuardThresholds(0.8, 0.55),
    )


async def _guard(
    adapted: str,
    *,
    use_case: GuardProbeUseCase | None = None,
    pack: ContextPack | None = None,
    timeout_s: float | None = 1.0,
) -> tuple[GuardOutcome, UsageLedger]:
    ledger = UsageLedger("r", 1.0)
    outcome = await (use_case or _use_case()).execute(
        adapted=adapted,
        approved=APPROVED,
        pack=pack or make_pack(),
        view=VIEW,
        ledger=ledger,
        timeout_s=timeout_s,
    )
    return outcome, ledger


async def test_a_light_adaptation_passes_with_one_embedding_call() -> None:
    adapted = "Kamu bilang gayanya habis. Kalau dorongannya habis, kenapa pesawat luar angkasa tetap melaju walau mesinnya mati?"
    outcome, ledger = await _guard(adapted)
    assert outcome == GuardOutcome(GuardResult.PASSED) and outcome.passed
    (record,) = ledger.records
    assert record.prompt_version == "embed.guard"


async def test_text_checks_run_first_and_cost_nothing() -> None:
    outcome, ledger = await _guard("Jawabanmu hampir benar. Kenapa pesawat tetap melaju?")
    assert outcome.result is GuardResult.BLOCKED_VERDICT
    assert ledger.records == ()


async def test_text_close_to_the_reference_is_blocked() -> None:
    pack = replace(make_pack(), answer_terms=("zzz",))
    adapted = "Kelereng melambat lalu berhenti karena ada gaya yang melawan gerak antara kelereng dan lantai?"
    outcome, _ = await _guard(
        adapted, use_case=_use_case(thresholds=GuardThresholds(0.5, 0.0)), pack=pack
    )
    assert outcome.result is GuardResult.BLOCKED_SIMILARITY


async def test_drift_from_the_approved_question_is_blocked() -> None:
    outcome, _ = await _guard("Apa makanan favoritmu di kantin sekolah?")
    assert outcome.result is GuardResult.BLOCKED_DRIFT


async def test_embedding_trouble_fails_closed() -> None:
    adapted = "Kalau dorongannya habis, kenapa pesawat luar angkasa tetap melaju?"
    outcome, ledger = await _guard(adapted, use_case=_use_case(FailingEmbedder()))
    assert outcome == GuardOutcome(GuardResult.NOT_RUN, degraded=True) and not outcome.passed
    (record,) = ledger.records
    assert record.status is CallStatus.ERROR
    outcome, ledger = await _guard(adapted, timeout_s=None)
    assert outcome == GuardOutcome(GuardResult.NOT_RUN, degraded=True)
    assert ledger.records == ()
