import pytest
from fastapi.testclient import TestClient

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import TransientLLMError
from tests.support.auth import AUTH
from tests.support.s3 import REFERENCE, by_prompt, choose_reply, classify_reply, pack_payload

URL = "/v1/s3/runs/warm"


@pytest.fixture
def llm(request: pytest.FixtureRequest) -> ScriptedLLM:
    """The two warm-up calls run concurrently, so each prompt gets its own reply queue."""
    return by_prompt(**request.param)


@pytest.mark.parametrize(
    "llm", [{"classify": [classify_reply()], "choose": [choose_reply()]}], indirect=True
)
def test_warm_up_calls_each_prompt_once(client: TestClient, llm: ScriptedLLM) -> None:
    response = client.post(URL, json={"context_pack": pack_payload()}, headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"] == {"warmed": True}
    assert sorted(i["purpose"] for i in payload["invocations"]) == ["probe_plan", "turn_analyze"]
    assert payload["warnings"] == []
    writer = next(r for r in llm.requests if r.system.startswith("You choose"))
    assert all(REFERENCE not in block.text for block in writer.blocks)


@pytest.mark.parametrize(
    "llm",
    [{"classify": [TransientLLMError("529")], "choose": [TransientLLMError("529")]}],
    indirect=True,
)
def test_a_failed_warm_up_is_only_a_warning(client: TestClient) -> None:
    response = client.post(URL, json={"context_pack": pack_payload()}, headers=AUTH)
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"] == {"warmed": False}
    assert payload["warnings"] == [
        "warm_failed:s3.classify_answer",
        "warm_failed:s3.choose_probe",
    ]
    assert len(payload["invocations"]) == 2
