import json
import uuid

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH


def test_dedupe_endpoint(client: TestClient, llm: ScriptedLLM) -> None:
    existing = str(uuid.uuid4())
    llm.queue(json.dumps({"same_as": "e1", "reason": "same idea"}))
    body = {
        "items": [
            {
                "key": "n1",
                "name": "Gaya gesek",
                "description": "Gaya yang melawan gerak.",
                "candidates": [
                    {
                        "concept_id": existing,
                        "name": "Gesekan",
                        "description": "",
                        "similarity": 0.87,
                    }
                ],
            }
        ]
    }
    response = client.post("/v1/s1/concepts/dedupe", json=body, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["result"]["decisions"] == [
        {
            "key": "n1",
            "action": "link",
            "existing_concept_id": existing,
            "basis": "judge",
            "similarity": 0.87,
            "reason": "same idea",
        }
    ]
    assert [i["purpose"] for i in response.json()["invocations"]] == ["kb_dedup"]
