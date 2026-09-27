import json
import uuid

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH


def test_align_cp_endpoint(client: TestClient, llm: ScriptedLLM) -> None:
    outcome = str(uuid.uuid4())
    llm.queue(json.dumps({"choice": "s1", "reason": "fits"}))
    body = {
        "items": [
            {
                "concept_ref": "5d0f7c1e-concept",
                "name": "Gaya gesek",
                "description": "Gaya yang melawan gerak.",
                "candidates": [
                    {
                        "outcome_id": outcome,
                        "element": "Pemahaman IPA",
                        "description": "Gaya dan gerak.",
                        "similarity": 0.7,
                    }
                ],
            }
        ]
    }
    response = client.post("/v1/s1/concepts/align-cp", json=body, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["result"]["alignments"] == [
        {
            "concept_ref": "5d0f7c1e-concept",
            "outcome_id": outcome,
            "basis": "judge",
            "reason": "fits",
        }
    ]
