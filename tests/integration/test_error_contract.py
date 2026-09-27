from fastapi.testclient import TestClient

from tests.support.auth import AUTH


def test_request_validation_errors_use_the_envelope(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"texts": [], "tag": "query"},
        headers={**AUTH, "X-Request-Id": "req-v"},
    )
    body = response.json()
    assert response.status_code == 422
    assert body["error"]["code"] == "invalid_input"
    assert body["error"]["details"]["errors"][0]["loc"] == ["body", "texts"]
    assert body["request_id"] == "req-v"
    assert body["invocations"] == []


def test_missing_service_key_uses_the_envelope(client: TestClient) -> None:
    response = client.post("/v1/embeddings", json={"texts": ["a"], "tag": "query"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
