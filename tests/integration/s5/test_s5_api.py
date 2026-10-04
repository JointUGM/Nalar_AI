from typing import Any

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from tests.support.auth import AUTH
from tests.support.s5 import (
    C_GESEK,
    C_LEMBAM,
    M_BERAT,
    M_DIAM,
    M_HABIS,
    SUMMARY,
    insight_reply,
    summary_reply,
)

INSIGHT_URL = "/v1/s5/class-insight"
SUMMARY_URL = "/v1/s5/parent-summaries/generate"
LEAKY = insight_reply(narrative="Ada 12 siswa yang bingung.")


def _misconception(id_: object, statement: str, count: int, resolved: int) -> dict[str, Any]:
    return {
        "misconception_id": str(id_),
        "statement": statement,
        "count": count,
        "resolved_count": resolved,
    }


def _insight_body(**changes: Any) -> dict[str, Any]:
    body = {
        "mission_title": "Gaya dan Gerak",
        "denominator": 28,
        "incomplete_count": 3,
        "concepts": [
            {
                "concept_id": str(C_GESEK),
                "name": "Gaya gesek",
                "mastered_count": 9,
                "developing_count": 7,
                "not_observed_count": 2,
                "misconceptions": [
                    _misconception(M_HABIS, "Gaya bisa habis seperti bensin", 10, 4)
                ],
            },
            {
                "concept_id": str(C_LEMBAM),
                "name": "Kelembaman",
                "mastered_count": 12,
                "developing_count": 8,
                "not_observed_count": 2,
                "misconceptions": [
                    _misconception(M_BERAT, "Benda berat selalu jatuh lebih cepat", 6, 1),
                    _misconception(M_DIAM, "Benda diam tidak punya gaya", 0, 0),
                ],
            },
        ],
    }
    return {**body, **changes}


def _summary_body(**changes: Any) -> dict[str, Any]:
    body = {
        "mission_title": "Gaya dan Gerak",
        "concepts": [
            {"name": "Gaya gesek", "outcome": "mastered"},
            {
                "name": "Kelembaman",
                "outcome": "misconception",
                "misconception_statement": "Benda berat selalu jatuh lebih cepat",
                "resolved_in_session": False,
            },
        ],
        "evaluation_summary": "Siswa menjelaskan gesekan dengan contoh lantai dan es.",
    }
    return {**body, **changes}


def test_class_insight_returns_placeholders_with_ids(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(
        insight_reply(suggestions=["Bahas {{count:m1}} siswa melalui pengamatan gaya gesek."])
    )
    response = client.post(
        INSIGHT_URL, json=_insight_body(), headers={**AUTH, "X-Request-Id": "s5-1"}
    )
    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert result["narrative"].startswith(f"{{{{count:{M_HABIS}}}}} dari {{{{total}}}} siswa")
    assert result["clusters"][0]["misconception_ids"] == [str(M_HABIS)]
    assert result["suggestions"] == [
        f"Bahas {{{{count:{M_HABIS}}}}} siswa melalui pengamatan gaya gesek."
    ]
    assert [i["purpose"] for i in payload["invocations"]] == ["class_map_insight"]
    assert payload["invocations"][0]["request_id"] == "s5-1"
    assert payload["warnings"] == []


def test_invalid_suggestions_fail_closed_with_both_invocations(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue(*[insight_reply(suggestions=["Ajak 12 siswa menjelaskan."])] * 2)
    response = client.post(INSIGHT_URL, json=_insight_body(), headers=AUTH)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_output_invalid"
    assert len(response.json()["invocations"]) == 2


def test_class_insight_reports_a_retry(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(LEAKY, insight_reply())
    payload = client.post(INSIGHT_URL, json=_insight_body(), headers=AUTH).json()
    assert payload["warnings"] == ["insight_retried"]


def test_class_insight_fails_with_502_and_both_invocations(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue(LEAKY, LEAKY)
    response = client.post(INSIGHT_URL, json=_insight_body(), headers=AUTH)
    assert response.status_code == 502
    payload = response.json()
    assert payload["error"]["code"] == "ai_output_invalid"
    assert [i["purpose"] for i in payload["invocations"]] == ["class_map_insight"] * 2


def test_class_insight_rejects_an_empty_class_and_inconsistent_counts(
    client: TestClient, llm: ScriptedLLM
) -> None:
    empty = client.post(INSIGHT_URL, json=_insight_body(denominator=0), headers=AUTH)
    assert empty.status_code == 422
    response = client.post(INSIGHT_URL, json=_insight_body(denominator=5), headers=AUTH)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert llm.requests == []


def test_parent_summary_returns_the_model_text(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(summary_reply())
    payload = client.post(SUMMARY_URL, json=_summary_body(), headers=AUTH).json()
    assert payload["result"] == {"content": SUMMARY, "source": "model"}
    assert [i["purpose"] for i in payload["invocations"]] == ["parent_summary"]
    assert payload["warnings"] == []


def test_parent_summary_falls_back_to_the_template(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(summary_reply("Nilai Ananda bagus."), summary_reply("Nilai Ananda bagus."))
    payload = client.post(SUMMARY_URL, json=_summary_body(), headers=AUTH).json()
    assert payload["result"]["source"] == "template"
    assert "Gaya gesek" in payload["result"]["content"]
    assert payload["warnings"] == ["summary_retried", "summary_template"]


def test_parent_summary_outage_is_a_503(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(TransientLLMError("503"), TransientLLMError("503"))
    response = client.post(SUMMARY_URL, json=_summary_body(), headers=AUTH)
    assert response.status_code == 503
    assert [i["status"] for i in response.json()["invocations"]] == ["error", "error"]


def test_parent_summary_needs_a_known_outcome(client: TestClient) -> None:
    body = _summary_body(concepts=[{"name": "Gaya gesek", "outcome": "excellent"}])
    assert client.post(SUMMARY_URL, json=body, headers=AUTH).status_code == 422


def test_the_service_key_is_required(client: TestClient) -> None:
    assert client.post(INSIGHT_URL, json=_insight_body()).status_code == 401
    assert client.post(SUMMARY_URL, json=_summary_body()).status_code == 401
