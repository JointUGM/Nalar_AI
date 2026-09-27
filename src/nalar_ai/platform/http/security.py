import hmac
from typing import Annotated

from fastapi import Header, HTTPException, Request, status


def require_service_key(
    request: Request, x_service_key: Annotated[str | None, Header()] = None
) -> None:
    """Only the NALAR backend may call this service (NFR-S10)."""
    expected: str = request.app.state.container.settings.service_key.get_secret_value()
    if not x_service_key or not hmac.compare_digest(x_service_key.encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid service key")
