from typing import Annotated, cast

from fastapi import Depends, Request

from nalar_ai.platform.http.dependencies import app_container, get_llm_gateway
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.subsystems.s4_session_evaluator.module import S4Module


def get_s4_module(request: Request) -> S4Module:
    return cast(S4Module, app_container(request).s4)


S4Dep = Annotated[S4Module, Depends(get_s4_module)]
GatewayDep = Annotated[LLMGateway, Depends(get_llm_gateway)]
