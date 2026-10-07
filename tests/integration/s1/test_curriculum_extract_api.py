import json

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH


def test_extract_returns_draft_and_invocations(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(json.dumps({"decree_code": None, "effective_on": None, "subjects": []}))
    body = {"title": "CP IPA", "pages": [{"page_number": 2, "text": "Tidak ada CP di sini."}]}
    response = client.post("/v1/s1/curriculum/extract", json=body, headers=AUTH)
    assert response.status_code == 200, response.text
    assert response.json()["result"]["subjects"] == []
    assert response.json()["invocations"][0]["purpose"] == "cp_extract"


def test_oversized_excerpt_is_rejected_without_a_model_call(
    client: TestClient, llm: ScriptedLLM
) -> None:
    pages = [{"page_number": n, "text": "x" * 40_000} for n in (1, 2)]
    response = client.post(
        "/v1/s1/curriculum/extract", json={"title": "CP", "pages": pages}, headers=AUTH
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert llm.requests == []


def test_duplicate_page_numbers_are_invalid(client: TestClient) -> None:
    pages = [{"page_number": 1, "text": "a"}, {"page_number": 1, "text": "b"}]
    response = client.post(
        "/v1/s1/curriculum/extract", json={"title": "CP", "pages": pages}, headers=AUTH
    )
    assert response.status_code == 422
