from fastapi.testclient import TestClient

from tests.support.auth import AUTH


def test_embeddings_endpoint_returns_vectors_and_provenance(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"texts": ["gaya gesek", "tekanan zat"], "tag": "query"},
        headers={**AUTH, "X-Request-Id": "req-e"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["embedding_model"] == "text-embedding-3-small"
    assert body["result"]["dimensions"] == 1536
    assert [len(v) for v in body["result"]["vectors"]] == [1536, 1536]
    (invocation,) = body["invocations"]
    assert invocation["purpose"] == "embedding"
    assert invocation["prompt_version"] == "embed.query"
    assert invocation["request_id"] == "req-e"
    assert invocation["provider"] == "fake"


def test_embeddings_endpoint_validates_input(client: TestClient) -> None:
    assert (
        client.post("/v1/embeddings", json={"texts": [], "tag": "query"}, headers=AUTH).status_code
        == 422
    )
    assert (
        client.post(
            "/v1/embeddings", json={"texts": ["a"], "tag": "nope"}, headers=AUTH
        ).status_code
        == 422
    )
    assert client.post("/v1/embeddings", json={"texts": ["a"], "tag": "query"}).status_code == 401
