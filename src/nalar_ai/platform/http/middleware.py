import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

REQUEST_ID_HEADER = "X-Request-Id"
# Room for multipart framing and form fields on top of the upload itself.
MULTIPART_SLACK_BYTES = 1024 * 1024
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


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Rejects oversized bodies from Content-Length before FastAPI parses (and spools) them."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        super().__init__(app)
        self._max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > self._max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": f"request body exceeds {self._max_body_bytes} bytes",
                        "details": {"max_bytes": self._max_body_bytes},
                    },
                    "request_id": getattr(request.state, "request_id", None),
                    "invocations": [],
                },
            )
        return await call_next(request)
