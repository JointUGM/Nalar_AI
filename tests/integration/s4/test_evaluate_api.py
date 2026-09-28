import json
from typing import Any

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from tests.support.auth import AUTH
from tests.support.s3 import M_HABIS, T_GESEK, T_LEMBAM, pack_payload
from tests.support.s4 import (
    TURN_IDS,
    make_turns,
    reflection_reply,
    rubric_payload,
    scoring_payload,
    scoring_reply,
    turns_payload,
)

URL = "/v1/s4/sessions/evaluate"


def _body(**changes: Any) -> dict[str, Any]:
    # The full S3 pack, question bank included: the backend sends the stored column unchanged.
    body = {"context_pack": pack_payload(), "rubric": rubric_payload(), "turns": turns_payload()}
    return {**body, **changes}


def test_evaluate_returns_rows_for_every_table_and_every_invocation(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue(scoring_reply(), reflection_reply())
    response = client.post(URL, json=_body(), headers={**AUTH, "X-Request-Id": "eval-1"})
    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert result["summary"].startswith("Siswa mengubah pendapat")
    assert [s["dimension"] for s in result["scores"]] == [
        "claim",
        "evidence",
        "mechanism",
        "transfer",
    ]
    assert result["scores"][0]["level"] == 3
    assert result["scores"][0]["evidence"] == [
        {"turn_id": str(TURN_IDS[2]), "quote": "ada gesekan jadi pelan-pelan berhenti"}
    ]
    assert result["concept_results"] == [
        {
            "concept_id": str(T_GESEK),
            "outcome": "mastered",
            "misconception_id": None,
            "evidence_turn_id": str(TURN_IDS[3]),
            "initial_misconception_id": None,
            "resolved_in_session": False,
        },
        {
            "concept_id": str(T_LEMBAM),
            "outcome": "developing",
            "misconception_id": None,
            "evidence_turn_id": str(TURN_IDS[1]),
            "initial_misconception_id": str(M_HABIS),
            "resolved_in_session": True,
        },
    ]
    assert result["turn_quality"][0] == {"turn_id": str(TURN_IDS[0]), "turn_index": 0, "quality": 1}
    assert result["reflection"]["source"] == "model"
    assert result["reflection"]["content"].count("\n\n") == 2
    assert [i["purpose"] for i in payload["invocations"]] == [
        "session_evaluation",
        "reflection_generation",
    ]
    assert {i["request_id"] for i in payload["invocations"]} == {"eval-1"}
    assert payload["warnings"] == []


def test_an_invalid_session_is_a_422_without_model_calls(
    client: TestClient, llm: ScriptedLLM
) -> None:
    response = client.post(URL, json=_body(turns=turns_payload(make_turns([""]))), headers=AUTH)
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_input"
    assert payload["error"]["details"] == {"problems": ["turns: no answered turn to evaluate"]}
    assert payload["invocations"] == []
    assert llm.requests == []


def test_a_rubric_level_missing_is_a_422(client: TestClient) -> None:
    rubric = rubric_payload()
    rubric["transfer"] = rubric["transfer"][:4]
    response = client.post(URL, json=_body(rubric=rubric), headers=AUTH)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_unverifiable_scores_fail_closed_and_return_the_paid_calls(
    client: TestClient, llm: ScriptedLLM
) -> None:
    invented = scoring_payload()
    invented["scores"][0]["evidence"] = [{"turn": "t2", "quote": "gaya gesek melawan gerak"}]
    llm.queue(json.dumps(invented), json.dumps(invented))
    response = client.post(URL, json=_body(), headers=AUTH)
    assert response.status_code == 502
    payload = response.json()
    assert payload["error"]["code"] == "ai_output_invalid"
    assert [i["purpose"] for i in payload["invocations"]] == ["session_evaluation"] * 2


def test_a_reflection_outage_fails_the_whole_evaluation_with_503(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue(scoring_reply(), TransientLLMError("503"), TransientLLMError("503"))
    response = client.post(URL, json=_body(), headers=AUTH)
    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "upstream_unavailable"
    assert [i["status"] for i in payload["invocations"]] == ["success", "error", "error"]


def test_the_service_key_is_required(client: TestClient) -> None:
    assert client.post(URL, json=_body()).status_code == 401


def test_s3_analysis_columns_sent_with_the_turns_are_ignored(
    client: TestClient, llm: ScriptedLLM
) -> None:
    turns = turns_payload()
    for turn in turns:
        turn.update(answer_state="misconception", detected_misconception_id=str(M_HABIS))
    llm.queue(scoring_reply(), reflection_reply())
    response = client.post(URL, json=_body(turns=turns), headers=AUTH)
    assert response.status_code == 200
    scorer_request = llm.requests[0]
    text = "".join(block.text for block in scorer_request.blocks)
    assert "answer_state" not in text and str(M_HABIS) not in text


def test_a_null_answer_from_the_database_counts_as_unanswered(
    client: TestClient, llm: ScriptedLLM
) -> None:
    # session_turns.answer_text is nullable: the probe shown when the time limit hit.
    turns = turns_payload()
    turns[4]["answer_text"] = None
    payload = scoring_payload()
    payload["turn_quality"] = payload["turn_quality"][:4]
    llm.queue(json.dumps(payload), reflection_reply())
    response = client.post(URL, json=_body(turns=turns), headers=AUTH)
    assert response.status_code == 200
    assert [q["turn_index"] for q in response.json()["result"]["turn_quality"]] == [0, 1, 2, 3]
