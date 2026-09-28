import json

from scripts.export_openapi import OPENAPI_PATH, render

REGENERATE = "run: uv run python scripts/export_openapi.py (then commit openapi.json)"


def test_openapi_json_is_committed_and_up_to_date() -> None:
    assert OPENAPI_PATH.exists(), REGENERATE
    assert OPENAPI_PATH.read_text(encoding="utf-8") == render(), (
        f"openapi.json is stale; {REGENERATE}"
    )


def test_every_v1_operation_requires_the_service_key() -> None:
    schema = json.loads(render())
    v1 = {path: ops for path, ops in schema["paths"].items() if path.startswith("/v1/")}
    assert len(v1) == 9
    for path, operations in v1.items():
        for operation in operations.values():
            names = {parameter["name"] for parameter in operation.get("parameters", [])}
            assert "x-service-key" in names, path


def test_every_v1_operation_documents_the_error_envelope() -> None:
    schema = json.loads(render())
    assert "ErrorEnvelope" in schema["components"]["schemas"]
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    for path, operations in schema["paths"].items():
        if not path.startswith("/v1/"):
            continue
        for operation in operations.values():
            for status in ("401", "422", "500", "502", "503"):
                ref = operation["responses"][status]["content"]["application/json"]["schema"][
                    "$ref"
                ]
                assert ref.endswith("/ErrorEnvelope"), (path, status)
