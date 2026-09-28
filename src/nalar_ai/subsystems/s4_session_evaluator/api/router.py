from fastapi import APIRouter, Depends

from nalar_ai.platform.http.envelope import ERROR_RESPONSES
from nalar_ai.platform.http.security import require_service_key
from nalar_ai.subsystems.s4_session_evaluator.api import evaluate

router = APIRouter(
    prefix="/v1/s4",
    tags=["s4"],
    dependencies=[Depends(require_service_key)],
    responses=ERROR_RESPONSES,
)
router.include_router(evaluate.router)
