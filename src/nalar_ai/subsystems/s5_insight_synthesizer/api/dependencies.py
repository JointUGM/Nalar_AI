from typing import Annotated, cast

from fastapi import Depends, Request

from nalar_ai.platform.http.dependencies import app_container, get_llm_gateway
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.subsystems.s5_insight_synthesizer.module import S5Module


def get_s5_module(request: Request) -> S5Module:
    return cast(S5Module, app_container(request).s5)


S5Dep = Annotated[S5Module, Depends(get_s5_module)]
GatewayDep = Annotated[LLMGateway, Depends(get_llm_gateway)]
