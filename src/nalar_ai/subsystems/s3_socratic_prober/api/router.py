from fastapi import APIRouter, Depends

from nalar_ai.platform.http.envelope import ERROR_RESPONSES
from nalar_ai.platform.http.security import require_service_key
from nalar_ai.subsystems.s3_socratic_prober.api import runs, turns

router = APIRouter(
    prefix="/v1/s3",
    tags=["s3"],
    dependencies=[Depends(require_service_key)],
    responses=ERROR_RESPONSES,
)
router.include_router(turns.router)
router.include_router(runs.router)
