import json
import uuid

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH

CHUNK_ID = str(uuid.uuid4())
BODY = {
    "subject": "IPA",
    "phase": "D",
    "section_title": "Bab 1 Gaya dan Gerak",
    "chunks": [
        {
            "id": CHUNK_ID,
            "heading_path": "Bab 1 > A. Gaya",
            "kind": "explanation",
            "content": "Gaya gesek melawan gerak benda.",
            "page_start": 4,
            "page_end": 4,
        }
    ],
}
REPLY = json.dumps(
    {
        "concepts": [
            {
                "key": "n1",
                "name": "Gaya gesek",
                "description": "Gaya yang melawan gerak benda.",
                "source_chunks": ["c1"],
                "prerequisites": [],
            }
        ],
        "existing_mentions": [],
    }
)


def test_extract_endpoint(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(REPLY)
    response = client.post("/v1/s1/concepts/extract", json=BODY, headers=AUTH)
    body = response.json()
    assert response.status_code == 200
    (concept,) = body["result"]["concepts"]
    assert concept["source_chunk_ids"] == [CHUNK_ID]
    assert len(concept["embedding"]) == 1536
    assert [i["purpose"] for i in body["invocations"]] == ["kb_extract", "embedding"]
    assert body["warnings"] == ["only 1 new concepts were extracted"]


def test_extract_endpoint_reports_invalid_output_with_invocations(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue("nonsense", "still nonsense")
    response = client.post("/v1/s1/concepts/extract", json=BODY, headers=AUTH)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_output_invalid"
    assert len(response.json()["invocations"]) == 2
