from fastapi import APIRouter, Depends

from nalar_ai.platform.http.envelope import ERROR_RESPONSES
from nalar_ai.platform.http.security import require_service_key
from nalar_ai.subsystems.s5_insight_synthesizer.api import class_insight, parent_summary

router = APIRouter(
    prefix="/v1/s5",
    tags=["s5"],
    dependencies=[Depends(require_service_key)],
    responses=ERROR_RESPONSES,
)
router.include_router(class_insight.router)
router.include_router(parent_summary.router)
