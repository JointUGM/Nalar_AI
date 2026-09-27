import json

from fastapi.testclient import TestClient

from tests.support.auth import AUTH
from tests.support.pdf_factory import Line, make_pdf

PDF = make_pdf(
    [
        [
            Line("Bab 1 Gaya dan Gerak", size=16, bold=True),
            Line("A. Gaya", size=14, bold=True, gap_before=6),
            "Gaya adalah tarikan atau dorongan pada benda.",
            Line("Uji Kompetensi", bold=True, gap_before=12),
            "1. Apa yang dimaksud dengan gaya?",
        ]
    ]
)


def _post(client: TestClient, spec: dict[str, object] | str) -> dict[str, object]:
    response = client.post(
        "/v1/s1/sections/chunk",
        files={"file": ("bab1.pdf", PDF, "application/pdf")},
        data={"spec": spec if isinstance(spec, str) else json.dumps(spec)},
        headers=AUTH,
    )
    return {"status": response.status_code, **response.json()}


def test_chunk_endpoint_returns_kind_labelled_embedded_chunks(client: TestClient) -> None:
    body = _post(client, {"page_start": 1, "page_end": 1, "title": "Bab 1 Gaya dan Gerak"})
    assert body["status"] == 200
    chunks = body["result"]["chunks"]  # type: ignore[index]
    assert [(c["local_index"], c["heading_path"], c["chunk_kind"]) for c in chunks] == [
        (0, "Bab 1 Gaya dan Gerak > A. Gaya", "explanation"),
        (1, "Bab 1 Gaya dan Gerak > A. Gaya > Uji Kompetensi", "exercise"),
    ]
    assert all(len(c["embedding"]) == 1536 and c["token_count"] > 0 for c in chunks)
    assert body["result"]["embedding_model"] == "text-embedding-3-small"  # type: ignore[index]
    assert body["invocations"][0]["prompt_version"] == "embed.chunk"  # type: ignore[index]


def test_chunk_endpoint_validates_the_spec(client: TestClient) -> None:
    assert _post(client, "not json")["error"]["code"] == "invalid_input"  # type: ignore[index]
    reversed_range = _post(client, {"page_start": 2, "page_end": 1, "title": "Bab 1"})
    assert reversed_range["status"] == 422
    outside = _post(client, {"page_start": 1, "page_end": 9, "title": "Bab 1"})
    assert outside["status"] == 422
