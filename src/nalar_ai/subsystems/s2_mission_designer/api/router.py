import asyncio

from fastapi import APIRouter, Depends, Request

from nalar_ai.platform.http.dependencies import LedgerDep, app_container
from nalar_ai.platform.http.envelope import ERROR_RESPONSES, Envelope, envelope
from nalar_ai.platform.http.security import require_service_key
from nalar_ai.shared.errors import UpstreamUnavailableError
from nalar_ai.subsystems.s2_mission_designer.application.design_mission import (
    DesignMissionUseCase,
    SelectTargetsUseCase,
)
from nalar_ai.subsystems.s2_mission_designer.application.revise_mission import ReviseMissionUseCase
from nalar_ai.subsystems.s2_mission_designer.application.schemas import (
    GenerateMissionIn,
    GenerateMissionOut,
    ReviseMissionIn,
    ReviseMissionOut,
    SelectTargetsIn,
    SelectTargetsOut,
)

router = APIRouter(
    prefix="/v1/s2",
    tags=["s2"],
    dependencies=[Depends(require_service_key)],
    responses=ERROR_RESPONSES,
)


@router.post("/targets/select", response_model=Envelope[SelectTargetsOut])
async def select_targets(
    body: SelectTargetsIn, ledger: LedgerDep, request: Request
) -> Envelope[SelectTargetsOut]:
    container = app_container(request)
    result = await SelectTargetsUseCase(container.llm, container.s2).execute(body, ledger)
    return envelope(result, ledger)


@router.post("/missions/generate", response_model=Envelope[GenerateMissionOut])
async def generate_mission(
    body: GenerateMissionIn, ledger: LedgerDep, request: Request
) -> Envelope[GenerateMissionOut]:
    container = app_container(request)
    result = await DesignMissionUseCase(container.llm, container.s2, container.tokens).execute(
        body, ledger
    )
    warnings = [f"ungrounded_concept:{i}" for i in result.ungrounded_concept_ids]
    return envelope(result, ledger, warnings)


@router.post("/missions/revise", response_model=Envelope[ReviseMissionOut])
async def revise_mission(
    body: ReviseMissionIn, ledger: LedgerDep, request: Request
) -> Envelope[ReviseMissionOut]:
    container = app_container(request)
    ledger.limit_invocations(container.settings.s2_revision_max_calls)
    try:
        async with asyncio.timeout(container.settings.s2_revision_timeout_seconds):
            result = await ReviseMissionUseCase(
                container.llm, container.s2, container.tokens
            ).execute(body, ledger)
    except TimeoutError as exc:
        raise UpstreamUnavailableError("mission revision deadline exceeded") from exc
    return envelope(
        result,
        ledger,
        [f"ungrounded_concept:{i}" for i in result.generation.ungrounded_concept_ids],
    )
