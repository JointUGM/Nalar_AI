from fastapi import UploadFile
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nalar_ai.container import Container
from nalar_ai.main import create_app
from nalar_ai.platform.http.dependencies import LedgerDep, SettingsDep
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.platform.http.uploads import read_pdf_upload
from nalar_ai.shared.enums import AiPurpose, CallStatus
from nalar_ai.shared.errors import OutputValidationError
from nalar_ai.shared.provenance import InvocationRecord, UsageLedger


class _Out(BaseModel):
    size: int


def _record(ledger: UsageLedger) -> InvocationRecord:
    return InvocationRecord(
        purpose=AiPurpose.KB_EXTRACT,
        model="claude-sonnet-5",
        prompt_version="s1.extract_concepts@v1",
        status=CallStatus.SUCCESS,
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        latency_ms=900,
        cost_usd=0.0007,
        request_id=ledger.request_id,
    )


def _app(container: Container) -> TestClient:
    app = create_app(container)

    @app.post("/_ok", response_model=Envelope[_Out])
    async def ok(ledger: LedgerDep) -> Envelope[_Out]:
        ledger.add(_record(ledger))
        return envelope(_Out(size=1), ledger, warnings=["short chapter"])

    @app.post("/_fail")
    async def fail(ledger: LedgerDep) -> None:
        ledger.add(_record(ledger))
        raise OutputValidationError("bad output", details={"errors": ["x"]})

    @app.post("/_upload", response_model=Envelope[_Out])
    async def upload(file: UploadFile, settings: SettingsDep, ledger: LedgerDep) -> Envelope[_Out]:
        assert settings.max_upload_bytes > 16
        data = await read_pdf_upload(file, max_bytes=16)
        return envelope(_Out(size=len(data)), ledger)

    return TestClient(app)


def test_success_envelope_lists_invocations_with_request_id(container: Container) -> None:
    client = _app(container)
    response = client.post("/_ok", headers={"X-Request-Id": "req-9"})
    body = response.json()
    assert response.status_code == 200
    assert body["result"] == {"size": 1}
    assert body["warnings"] == ["short chapter"]
    assert body["invocations"][0]["request_id"] == "req-9"
    assert body["invocations"][0]["purpose"] == "kb_extract"
    assert body["invocations"][0]["retrieval"] == []


def test_error_envelope_carries_invocations_spent(container: Container) -> None:
    client = _app(container)
    response = client.post("/_fail", headers={"X-Request-Id": "req-10"})
    body = response.json()
    assert response.status_code == 502
    assert body["error"] == {
        "code": "ai_output_invalid",
        "message": "bad output",
        "details": {"errors": ["x"]},
    }
    assert body["request_id"] == "req-10"
    assert len(body["invocations"]) == 1


def test_pdf_upload_checks(container: Container) -> None:
    client = _app(container)
    ok = client.post("/_upload", files={"file": ("a.pdf", b"%PDF-1.7 tiny", "application/pdf")})
    assert ok.status_code == 200
    assert ok.json()["result"] == {"size": 13}
    not_pdf = client.post("/_upload", files={"file": ("a.pdf", b"hello", "application/pdf")})
    assert not_pdf.status_code == 422
    assert not_pdf.json()["error"]["code"] == "invalid_input"
    too_big = client.post(
        "/_upload", files={"file": ("a.pdf", b"%PDF-" + b"x" * 64, "application/pdf")}
    )
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "payload_too_large"
