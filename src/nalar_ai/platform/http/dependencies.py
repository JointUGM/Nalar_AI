from typing import Annotated, Any, cast

from fastapi import Depends, Request

from nalar_ai.settings import Settings
from nalar_ai.shared.provenance import UsageLedger


def app_container(request: Request) -> Any:
    """The composition root, typed Any here so platform code never imports nalar_ai.container."""
    return request.app.state.container


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, app_container(request).settings)


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
