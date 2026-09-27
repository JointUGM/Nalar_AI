from fastapi import APIRouter, Depends

from nalar_ai.platform.http.security import require_service_key
from nalar_ai.subsystems.s1_knowledge_base.api import sections

router = APIRouter(prefix="/v1/s1", tags=["s1"], dependencies=[Depends(require_service_key)])
router.include_router(sections.router)
