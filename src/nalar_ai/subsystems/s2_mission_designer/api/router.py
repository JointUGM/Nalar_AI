from fastapi import APIRouter, Depends, Request

from nalar_ai.platform.http.dependencies import LedgerDep, app_container
from nalar_ai.platform.http.envelope import ERROR_RESPONSES, Envelope, envelope
from nalar_ai.platform.http.security import require_service_key
from nalar_ai.subsystems.s2_mission_designer.application.design_mission import (
    DesignMissionUseCase,
    SelectTargetsUseCase,
)
from nalar_ai.subsystems.s2_mission_designer.application.schemas import (
    GenerateMissionIn,
    GenerateMissionOut,
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
