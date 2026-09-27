from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from nalar_ai import __version__

router = APIRouter(tags=["health"])


class HealthOut(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    return HealthOut(status="ok", service="nalar-ai", version=__version__)
