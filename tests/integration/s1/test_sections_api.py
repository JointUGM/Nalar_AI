from fastapi.testclient import TestClient

from tests.support.auth import AUTH
from tests.support.pdf_factory import Line, ScannedPage, make_pdf

BODY = "Gaya adalah tarikan atau dorongan yang dapat mengubah gerak benda."


def test_detect_sections(client: TestClient) -> None:
    pdf = make_pdf(
        [[Line("Bab 1 Gaya dan Gerak", size=16, bold=True), BODY], ScannedPage(), ["Bab 2", BODY]],
        toc=[(1, "Bab 1 Gaya dan Gerak", 1), (1, "Bab 2 Tekanan Zat", 3)],
    )
    response = client.post(
        "/v1/s1/sections/detect",
        files={"file": ("buku.pdf", pdf, "application/pdf")},
        data={"fallback_title": "Buku IPA"},
        headers=AUTH,
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["page_count"] == 3
    assert body["result"]["has_toc"] is True
    assert body["result"]["pages_without_text"] == [2]
    assert [(s["title"], s["page_start"], s["page_end"]) for s in body["result"]["sections"]] == [
        ("Bab 1 Gaya dan Gerak", 1, 3),
        ("Bab 2 Tekanan Zat", 3, 3),
    ]
    assert body["invocations"] == []


def test_detect_sections_rejects_bad_input(client: TestClient) -> None:
    files = {"file": ("x.pdf", b"hello", "application/pdf")}
    not_pdf = client.post("/v1/s1/sections/detect", files=files, headers=AUTH)
    assert not_pdf.status_code == 422 and not_pdf.json()["error"]["code"] == "invalid_input"
    broken = {"file": ("x.pdf", b"%PDF-1.7 broken", "application/pdf")}
    unreadable = client.post("/v1/s1/sections/detect", files=broken, headers=AUTH)
    assert unreadable.status_code == 422
    assert unreadable.json()["error"]["code"] == "document_unreadable"
    assert client.post("/v1/s1/sections/detect", files=files).status_code == 401
