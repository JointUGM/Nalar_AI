from fastapi import UploadFile
from fastapi.testclient import TestClient
from pydantic import SecretStr

from nalar_ai.container import build_container
from nalar_ai.main import create_app
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.settings import Settings
from tests.support.auth import SERVICE_KEY


def _client(max_upload_bytes: int) -> TestClient:
    settings = Settings.model_construct(
        service_key=SecretStr(SERVICE_KEY), llm_provider="fake", max_upload_bytes=max_upload_bytes
    )
    app = create_app(
        build_container(settings, llm_port=ScriptedLLM(), embedding_port=HashingEmbedder())
    )

    @app.post("/_upload")
    async def upload(file: UploadFile) -> dict[str, bool]:
        # Teacher material must never touch disk (design doc §15).
        return {"on_disk": bool(getattr(file.file, "_rolled", False))}

    return TestClient(app)


def test_large_pdf_uploads_stay_in_memory() -> None:
    body = b"%PDF-1.7 " + b"x" * (3 * 1024 * 1024)
    response = _client(50 * 1024 * 1024).post(
        "/_upload", files={"file": ("big.pdf", body, "application/pdf")}
    )
    assert response.status_code == 200
    assert response.json() == {"on_disk": False}


def test_oversized_bodies_are_rejected_before_parsing() -> None:
    body = b"%PDF-1.7 " + b"x" * (3 * 1024 * 1024)
    response = _client(1000).post("/_upload", files={"file": ("big.pdf", body, "application/pdf")})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
