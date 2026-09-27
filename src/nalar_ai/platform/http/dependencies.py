from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import Depends, Request

from nalar_ai.settings import Settings
from nalar_ai.shared.provenance import UsageLedger

if TYPE_CHECKING:
    from nalar_ai.platform.embeddings.service import EmbeddingService
    from nalar_ai.platform.llm.gateway import LLMGateway
    from nalar_ai.platform.tokens import TokenCounter


def app_container(request: Request) -> Any:
    """The composition root, typed Any here so platform code never imports nalar_ai.container."""
    return request.app.state.container


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, app_container(request).settings)


def get_embedding_service(request: Request) -> "EmbeddingService":
    return cast("EmbeddingService", app_container(request).embeddings)


def get_llm_gateway(request: Request) -> "LLMGateway":
    return cast("LLMGateway", app_container(request).llm)


def get_token_counter(request: Request) -> "TokenCounter":
    return cast("TokenCounter", app_container(request).tokens)


def get_ledger(request: Request) -> UsageLedger:
    """One ledger per request. The error handler reads it back from request.state."""
    ledger = UsageLedger(
        request_id=request.state.request_id,
        max_cost_usd=get_settings_dep(request).max_cost_usd_per_request,
    )
    request.state.ledger = ledger
    return ledger


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LedgerDep = Annotated[UsageLedger, Depends(get_ledger)]
