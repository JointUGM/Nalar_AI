from typing import Any

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH
from tests.support.s3 import (
    M_HABIS,
    T_GESEK,
    T_LEMBAM,
    anchor,
    choose_reply,
    classify_reply,
    history_payload,
    pack_payload,
)

URL = "/v1/s3/turns/next"


def _body(
    answer: str = "kelerengnya berhenti karena gayanya habis", **changes: Any
) -> dict[str, Any]:
    body = {
        "planner_mode": "hybrid",
        "context_pack": pack_payload(),
        "history": history_payload([anchor(answer)]),
        "elapsed_seconds": 60,
    }
    return {**body, **changes}


def test_next_turn_returns_a_guarded_probe_and_every_invocation(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue(classify_reply(), choose_reply())
    response = client.post(URL, json=_body(), headers={**AUTH, "X-Request-Id": "turn-1"})
    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert result["action"] == "probe"
    assert result["analysis"] == {
        "answer_type": "misconception",
        "source": "model",
        "misconception_id": str(M_HABIS),
        "secondary_misconception_id": None,
        "frustration": False,
        "key_phrase": None,
    }
    probe = result["probe"]
    assert probe["move"] == "counter_example"
    assert probe["allowed_moves"] == ["counter_example", "explain_mechanism", "decompose"]
    assert probe["target_concept_id"] == str(T_LEMBAM)
    assert probe["question_bank_id"] == "lembam-counter_example"
    assert probe["question_source"] == "adapted"
    assert probe["move_source"] == "planner"
    assert probe["reason_code"] == "default"
    assert probe["guard_result"] == "passed"
    assert result["end_reason"] is None and result["safety_message"] is None
    assert result["coverage"] == [
        {"concept_id": str(T_GESEK), "challenged": False},
        {"concept_id": str(T_LEMBAM), "challenged": True},
    ]
    assert [i["purpose"] for i in payload["invocations"]] == [
        "turn_analyze",
        "probe_plan",
        "embedding",
    ]
    assert {i["request_id"] for i in payload["invocations"]} == {"turn-1"}
    assert payload["warnings"] == []


def test_model_trouble_still_returns_a_probe_with_warnings(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue("not json", "not json either")
    response = client.post(URL, json=_body(), headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["probe"]["move_source"] == "fallback_invalid"
    assert payload["result"]["probe"]["question_source"] == "approved"
    assert payload["warnings"] == ["classification_fallback", "choice_fallback"]
    assert len(payload["invocations"]) == 2


def test_distress_returns_a_safety_pause(client: TestClient, llm: ScriptedLLM) -> None:
    response = client.post(URL, json=_body("aku pengen mati aja"), headers=AUTH)
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["action"] == "safety_pause"
    assert result["safety_message"].startswith("Terima kasih")
    assert result["probe"] is None
    assert llm.requests == []


def test_an_unusable_pack_is_a_422_with_problems(client: TestClient) -> None:
    pack = pack_payload()
    pack["question_bank"] = [q for q in pack["question_bank"] if q["id"] != "gesek-transfer"]
    response = client.post(URL, json=_body(context_pack=pack), headers=AUTH)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_input"
    assert error["details"]["problems"] == [
        f"target {T_GESEK} has no approved question for: transfer"
    ]


def test_schema_errors_use_the_error_envelope(client: TestClient) -> None:
    response = client.post(URL, json=_body(history=[]), headers=AUTH)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_the_service_key_is_required(client: TestClient) -> None:
    assert client.post(URL, json=_body()).status_code == 401
