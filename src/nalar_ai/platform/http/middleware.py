import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-Id"
_MAX_REQUEST_ID_LENGTH = 128


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Accepts the backend's X-Request-Id (or makes one) so a call can be traced across services."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        valid = 0 < len(incoming) <= _MAX_REQUEST_ID_LENGTH and incoming.isprintable()
        request_id = incoming if valid else uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
