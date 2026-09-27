import json
import uuid
from collections.abc import Sequence

from fastapi.testclient import TestClient

from nalar_ai.container import build_container
from nalar_ai.main import create_app
from nalar_ai.platform.embeddings.ports import EmbeddingBatch, TransientEmbeddingError
from nalar_ai.platform.http.dependencies import LedgerDep
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.settings import Settings
from nalar_ai.shared.enums import AiPurpose, CallStatus
from nalar_ai.shared.provenance import InvocationRecord
from tests.support.auth import AUTH


class _DownEmbedder:
    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        raise TransientEmbeddingError("503")


def test_paid_extraction_survives_in_the_envelope_when_embeddings_are_down(
    settings: Settings,
) -> None:
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "concepts": [
                        {
                            "key": "n1",
                            "name": "Gaya gesek",
                            "description": "Melawan gerak.",
                            "source_chunks": ["c1"],
                            "prerequisites": [],
                        }
                    ],
                    "existing_mentions": [],
                }
            )
        ]
    )
    app = create_app(build_container(settings, llm_port=llm, embedding_port=_DownEmbedder()))
    body = {
        "subject": "IPA",
        "phase": "D",
        "section_title": "Bab 1",
        "chunks": [
            {"id": str(uuid.uuid4()), "kind": "explanation", "content": "Gaya gesek melawan gerak."}
        ],
    }
    with TestClient(app) as client:
        response = client.post("/v1/s1/concepts/extract", json=body, headers=AUTH)
    assert response.status_code == 503
    records = response.json()["invocations"]
    assert [r["purpose"] for r in records] == ["kb_extract", "embedding", "embedding", "embedding"]
    assert records[0]["status"] == "success"


def test_unhandled_errors_render_the_envelope_with_invocations(settings: Settings) -> None:
    app = create_app(build_container(settings, llm_port=ScriptedLLM()))

    @app.post("/_boom")
    async def boom(ledger: LedgerDep) -> None:
        ledger.add(
            InvocationRecord(
                purpose=AiPurpose.KB_EXTRACT,
                model="claude-sonnet-5",
                prompt_version="s1.extract_concepts@v1",
                status=CallStatus.SUCCESS,
                input_tokens=1,
                output_tokens=1,
                cache_read_tokens=0,
                cache_write_tokens=0,
                latency_ms=1,
                cost_usd=0.001,
                request_id=ledger.request_id,
            )
        )
        raise RuntimeError("bug")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/_boom", headers={"X-Request-Id": "req-boom"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["request_id"] == "req-boom"
    assert len(body["invocations"]) == 1
