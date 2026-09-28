from fastapi import APIRouter
from pydantic import BaseModel

from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.subsystems.s3_socratic_prober.api.dependencies import GatewayDep, S3Dep
from nalar_ai.subsystems.s3_socratic_prober.api.pack import ContextPackIn
from nalar_ai.subsystems.s3_socratic_prober.application.warm_cache import WarmCacheUseCase

router = APIRouter()


class WarmIn(BaseModel):
    context_pack: ContextPackIn


class WarmOut(BaseModel):
    warmed: bool  # false when any prompt could not be warmed (see warnings)


@router.post("/runs/warm", response_model=Envelope[WarmOut])
async def warm(body: WarmIn, ledger: LedgerDep, llm: GatewayDep, s3: S3Dep) -> Envelope[WarmOut]:
    """Warm the prompt cache when the teacher starts a run (TC-7). Call it once per run.

    A failed warm-up is not an error: the run works without it, only the first turns are
    slower. Failures come back as warnings.
    """
    warnings = await WarmCacheUseCase(llm=llm, policy=s3.policy).execute(
        body.context_pack.to_pack(), ledger
    )
    return envelope(WarmOut(warmed=not warnings), ledger, warnings)
