import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nalar_ai.platform.http.envelope import InvocationOut
from nalar_ai.shared.errors import NalarAIError
from nalar_ai.shared.provenance import UsageLedger

logger = logging.getLogger("nalar_ai")


def error_response(
    request: Request, status_code: int, code: str, message: str, details: dict[str, Any]
) -> JSONResponse:
    """The error envelope, carrying every invocation the request already paid for (AI-6)."""
    ledger: UsageLedger | None = getattr(request.state, "ledger", None)
    records = ledger.records if ledger is not None else ()
    request_id = getattr(request.state, "request_id", None)
    logger.warning(
        "request failed code=%s request_id=%s invocations=%d", code, request_id, len(records)
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message, "details": details},
            "request_id": request_id,
            "invocations": [
                InvocationOut.from_record(record).model_dump(mode="json") for record in records
            ],
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NalarAIError)
    async def handle_nalar_error(request: Request, exc: NalarAIError) -> JSONResponse:
        return error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", exc_info=exc)
        return error_response(request, 500, "internal_error", "internal error", {})
