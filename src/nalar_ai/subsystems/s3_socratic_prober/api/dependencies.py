from typing import Annotated, cast

from fastapi import Depends, Request

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.http.dependencies import (
    app_container,
    get_embedding_service,
    get_llm_gateway,
)
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.subsystems.s3_socratic_prober.module import S3Module


def get_s3_module(request: Request) -> S3Module:
    return cast(S3Module, app_container(request).s3)


S3Dep = Annotated[S3Module, Depends(get_s3_module)]
EmbeddingsDep = Annotated[EmbeddingService, Depends(get_embedding_service)]
GatewayDep = Annotated[LLMGateway, Depends(get_llm_gateway)]
