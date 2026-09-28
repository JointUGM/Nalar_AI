from pathlib import Path

import pytest

from nalar_ai.container import PROMPT_DIRECTORIES, build_container
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.http.envelope import InvocationOut
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.prompts.registry import PromptRegistry, parse_prompt
from nalar_ai.settings import Settings
from nalar_ai.shared.enums import AiPurpose, CallStatus, EvalPurpose
from nalar_ai.shared.provenance import InvocationRecord

EVAL_PROMPT = """---
id: eval.judge
version: 1
purpose: eval_leak_judge
tier: judge
max_tokens: 100
---
=== system ===
You judge.
=== task ===
Judge {{text}}.
"""


def _record(purpose: AiPurpose | EvalPurpose) -> InvocationRecord:
    return InvocationRecord(
        purpose=purpose,
        model="claude-opus-5",
        prompt_version="eval.judge@v1",
        status=CallStatus.SUCCESS,
        input_tokens=1,
        output_tokens=1,
        cache_read_tokens=0,
        cache_write_tokens=0,
        latency_ms=1,
        cost_usd=0.0,
        request_id="r",
    )


def test_eval_prompts_parse_with_their_own_purpose() -> None:
    assert parse_prompt(EVAL_PROMPT).purpose is EvalPurpose.LEAK_JUDGE


def test_eval_invocations_never_reach_an_envelope() -> None:
    assert InvocationOut.from_record(_record(AiPurpose.TURN_ANALYZE)).purpose is (
        AiPurpose.TURN_ANALYZE
    )
    with pytest.raises(ValueError, match="eval-only"):
        InvocationOut.from_record(_record(EvalPurpose.LEAK_JUDGE))


def test_no_shipped_prompt_uses_an_eval_purpose() -> None:
    templates = PromptRegistry.from_directories(PROMPT_DIRECTORIES).templates()
    assert templates
    assert not any(isinstance(t.purpose, EvalPurpose) for t in templates)


def test_evals_can_add_their_own_prompt_directory(tmp_path: Path, settings: Settings) -> None:
    (tmp_path / "judge.v1.prompt").write_text(EVAL_PROMPT, encoding="utf-8")
    container = build_container(
        settings,
        llm_port=ScriptedLLM(),
        embedding_port=HashingEmbedder(),
        extra_prompt_directories=(tmp_path,),
    )
    assert "eval.judge" in container.prompts.ids()
