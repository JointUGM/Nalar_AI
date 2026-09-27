import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nalar_ai.platform.http.envelope import InvocationOut
from nalar_ai.shared.errors import NalarAIError
from nalar_ai.shared.provenance import UsageLedger

logger = logging.getLogger("nalar_ai")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(NalarAIError)
    async def handle_nalar_error(request: Request, exc: NalarAIError) -> JSONResponse:
        ledger: UsageLedger | None = getattr(request.state, "ledger", None)
        records = ledger.records if ledger is not None else ()
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "request failed code=%s request_id=%s invocations=%d",
            exc.code,
            request_id,
            len(records),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {"code": exc.code, "message": exc.message, "details": exc.details},
                "request_id": request_id,
                "invocations": [
                    InvocationOut.from_record(record).model_dump(mode="json") for record in records
                ],
            },
        )
