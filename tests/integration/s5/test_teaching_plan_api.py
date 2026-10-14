import json
from typing import Any

from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from tests.support.auth import AUTH
from tests.support.s5 import M_BERAT

URL = "/v1/s5/teaching-plan"
EXAMPLE = "Bola dan kertas kusut dijatuhkan bersamaan dari ketinggian bahu."
BODY: dict[str, Any] = {
    "mission_title": "Gaya dan Gerak",
    "learning_objective": "Menjelaskan gerak jatuh benda.",
    "concept_name": "Kelembaman",
    "misconception_id": str(M_BERAT),
    "statement": "Benda berat selalu jatuh lebih cepat",
    "correct_understanding": "Tanpa hambatan udara, benda jatuh dengan percepatan yang sama.",
    "counter_examples": [EXAMPLE],
    "questions": [{"id": "q1", "text": "Mengapa bola dan kertas sampai bersamaan?"}],
}


def _reply(**changes: Any) -> str:
    return json.dumps(
        {
            "objective": "Siswa memahami bahwa berat tidak menentukan kecepatan jatuh.",
            "steps": [
                {
                    "phase": "elicit",
                    "teacher_action": "Tunjukkan bola dan kertas.",
                    "ask": "Mana duluan?",
                },
                {
                    "phase": "contrast",
                    "teacher_action": "Jatuhkan bersamaan.",
                    "ask": "Apa yang terlihat?",
                },
                {
                    "phase": "explain",
                    "teacher_action": "Sebut gagasan keliru.",
                    "ask": "Jelaskan lagi.",
                },
            ],
            "counter_example": 1,
            "check_question": "Batu besar dan kecil dijatuhkan, mana yang sampai dulu?",
            "look_for": "Siswa menyebut hambatan udara, bukan berat.",
            "question_ids": ["q1"],
            **changes,
        }
    )


def test_teaching_plan_copies_the_reviewed_example(client: TestClient, llm: ScriptedLLM) -> None:
    llm.queue(_reply())
    payload = client.post(URL, json=BODY, headers=AUTH).json()
    plan = payload["result"]
    assert [s["phase"] for s in plan["steps"]] == ["elicit", "contrast", "explain"]
    assert plan["counter_example"] == EXAMPLE
    assert plan["misconception_id"] == str(M_BERAT)
    assert [i["purpose"] for i in payload["invocations"]] == ["teaching_plan"]
    assert "<idea>" in str(llm.requests[0])  # teacher text is fenced


def test_teaching_plan_fails_closed_with_every_invocation(
    client: TestClient, llm: ScriptedLLM
) -> None:
    llm.queue(_reply(objective="Kebanyakan siswa keliru."), _reply(counter_example=4))
    response = client.post(URL, json=BODY, headers=AUTH)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_output_invalid"
    assert len(response.json()["invocations"]) == 2
