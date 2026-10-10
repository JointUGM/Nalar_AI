import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from nalar_ai import __version__
from nalar_ai.container import Container
from nalar_ai.main import create_app
from nalar_ai.platform.http.security import require_service_key
from tests.support.auth import AUTH


def test_health_is_public_and_reports_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "nalar-ai", "version": __version__}


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-Id": "req-123"})
    assert response.headers["X-Request-Id"] == "req-123"


def test_request_id_is_generated_when_missing(client: TestClient) -> None:
    response = client.get("/health")
    assert len(response.headers["X-Request-Id"]) == 32


def test_service_key_is_required_on_protected_routes(container: Container) -> None:
    app = create_app(container)

    @app.get("/_protected", dependencies=[Depends(require_service_key)])
    async def protected() -> dict[str, str]:
        return {"ok": "yes"}

    with TestClient(app) as test_client:
        assert test_client.get("/_protected").status_code == 401
        assert test_client.get("/_protected", headers={"X-Service-Key": "wrong"}).status_code == 401
        assert test_client.get("/_protected", headers=AUTH).status_code == 200


@pytest.mark.parametrize("key", [None, "invalid"])
def test_every_service_route_rejects_unauthenticated_requests(
    client: TestClient, key: str | None
) -> None:
    assert isinstance(client.app, FastAPI)
    paths = client.app.openapi()["paths"]
    protected = [path for path in paths if path.startswith("/v1/")]
    assert len(protected) == 16
    for path in protected:
        headers = {} if key is None else {"X-Service-Key": key}
        response = client.post(path, headers=headers, json={})
        assert response.status_code == 401, path
        assert response.json()["error"]["code"] == "unauthorized", path
        assert response.json()["invocations"] == [], path
