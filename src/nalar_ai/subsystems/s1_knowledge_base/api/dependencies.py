from typing import Annotated, cast

from fastapi import Depends, Request

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.http.dependencies import (
    app_container,
    get_embedding_service,
    get_llm_gateway,
    get_token_counter,
)
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.tokens import TokenCounter
from nalar_ai.subsystems.s1_knowledge_base.module import S1Module


def get_s1_module(request: Request) -> S1Module:
    return cast(S1Module, app_container(request).s1)


S1Dep = Annotated[S1Module, Depends(get_s1_module)]
EmbeddingsDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
GatewayDep = Annotated[LLMGateway, Depends(get_llm_gateway)]
TokensDep = Annotated[TokenCounter, Depends(get_token_counter)]
