import json
import uuid

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH


def test_misconceptions_endpoint(client: TestClient, llm: ScriptedLLM) -> None:
    chunk = str(uuid.uuid4())
    llm.queue(
        json.dumps(
            {
                "misconceptions": [
                    {
                        "statement": "Benda berhenti karena gaya dorongnya habis",
                        "correct_understanding": "Benda melambat karena ada gaya gesek.",
                        "student_phrasings": ["gayanya habis", "tenaganya hilang"],
                        "counter_examples": ["Mengapa pesawat luar angkasa tetap melaju?"],
                        "source_chunks": ["c1"],
                        "library": None,
                    }
                ]
            }
        )
    )
    body = {
        "subject": "IPA",
        "phase": "D",
        "chunks": [
            {
                "id": chunk,
                "heading_path": "Bab 1",
                "kind": "explanation",
                "content": "Gaya gesek melawan gerak.",
            }
        ],
        "concepts": [
            {
                "concept_ref": "n1",
                "name": "Gaya gesek",
                "description": "Melawan gerak.",
                "source_chunk_ids": [chunk],
            }
        ],
    }
    response = client.post("/v1/s1/misconceptions/generate", json=body, headers=AUTH)
    result = response.json()["result"]
    assert response.status_code == 200
    (item,) = result["misconceptions"]
    assert item["source_chunk_ids"] == [chunk] and item["library_id"] is None
    assert item["detection_cues"] == ["gayanya habis", "tenaganya hilang"]
    assert result["failed"] == [] and result["embedding_model"] == "text-embedding-3-small"
    assert [i["purpose"] for i in response.json()["invocations"]] == [
        "kb_misconceptions",
        "embedding",
    ]
