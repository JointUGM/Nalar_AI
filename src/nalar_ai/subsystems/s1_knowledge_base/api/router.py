from fastapi import APIRouter, Depends

from nalar_ai.platform.http.envelope import ERROR_RESPONSES
from nalar_ai.platform.http.security import require_service_key
from nalar_ai.subsystems.s1_knowledge_base.api import (
    align_cp,
    dedupe,
    extract,
    misconceptions,
    sections,
)

router = APIRouter(
    prefix="/v1/s1",
    tags=["s1"],
    dependencies=[Depends(require_service_key)],
    responses=ERROR_RESPONSES,
)
router.include_router(sections.router)
router.include_router(extract.router)
router.include_router(dedupe.router)
router.include_router(align_cp.router)
router.include_router(misconceptions.router)
