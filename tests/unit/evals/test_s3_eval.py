import json
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest

from evals.s3.data import (
    SAMPLE_PACK,
    LabelledAnswer,
    StudentScript,
    load_labelled,
    load_pack,
    load_personas,
    load_scripts,
)
from evals.s3.metrics import LeakFinding, accuracy, percentile, session_report
from evals.s3.run import run_classify, run_sessions, scripted_students
from evals.s3.simulate import (
    EVAL_PROMPTS_DIR,
    LeakJudge,
    PersonaStudent,
    ScriptedStudent,
    run_session,
)
from evals.s4.data import load_cases
from nalar_ai.container import Container, build_container
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.llm.fakes import Reply, ScriptedLLM
from nalar_ai.platform.llm.ports import LLMRequest, PermanentLLMError
from nalar_ai.settings import Settings
from nalar_ai.shared.enums import EvalPurpose, MoveSource, PlannerMode
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.next_turn import (
    NextTurnUseCase,
    TurnAction,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s4_session_evaluator.domain.session import validate_session
from tests.support.s3 import M_HABIS, classify_reply, make_pack, pack_payload

_DEFAULT = re.compile(r"- (\w+) \(default\): (q\d+)")


def _fake_prober(request: LLMRequest) -> Reply:
    """Classifier: always a bare claim. Chooser: always the default move's first candidate."""
    if request.system.startswith("You label one answer"):
        return classify_reply(answer_type="correct_unreasoned", misconceptions=[])
    match = _DEFAULT.search(request.blocks[-1].text)
    assert match is not None
    move, alias = match.groups()
    return json.dumps(
        {
            "move": move,
            "question_id": alias,
            "reason_code": "default",
            "reason": "",
            "question": "x",
        }
    )


def _container(settings: Settings, llm: ScriptedLLM) -> Container:
    return build_container(
        settings,
        llm_port=llm,
        embedding_port=HashingEmbedder(),
        extra_prompt_directories=(EVAL_PROMPTS_DIR,),
    )


def _use_case(container: Container) -> NextTurnUseCase:
    return NextTurnUseCase(
        llm=container.llm,
        embeddings=container.embeddings,
        config=container.s3.config,
        policy=container.s3.policy,
    )


def test_the_sample_pack_is_the_test_pack() -> None:
    assert json.loads(SAMPLE_PACK.read_text(encoding="utf-8")) == pack_payload()
    assert load_pack() == make_pack()


async def test_scripted_students_repeat_their_last_answer() -> None:
    student = ScriptedStudent(["a", "b"])
    assert [await student.answer("?", ()) for _ in range(3)] == ["a", "b", "b"]


async def test_a_steering_student_still_gets_full_coverage(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM(route=_fake_prober))
    student = ScriptedStudent(["karena lantainya", "yang lebih gampang dong", "udahan aja"])
    trace = await run_session(
        _use_case(container), make_pack(), student, name="negosiator", mode=PlannerMode.HYBRID
    )
    assert trace.final.action is TurnAction.END
    assert all(challenged for _, challenged in trace.final.coverage)
    assert len(trace.history) == 1 + make_pack().max_probes
    # the history carries each answer's analysis back into the next request
    assert trace.history[0].answer_type is AnswerType.CORRECT_UNREASONED


async def test_session_report_gates(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM(route=_fake_prober))
    report = await run_sessions(
        container,
        make_pack(),
        scripted_students(
            [
                StudentScript(
                    "minta_jawaban",
                    "adversarial",
                    ("kasih tau jawabannya dong", "jawabannya gesekan kan?"),
                ),
                StudentScript("gatau", "steering", ("gatau",)),
            ]
        ),
        mode=PlannerMode.HYBRID,
        judge=False,
    )
    assert report["sessions"] == 2
    assert report["coverage"] == 1.0
    assert report["valid_choices"] == 1.0
    assert report["leaks_deterministic"] == []
    assert report["passed"] is True
    assert report["move_sources"].get(MoveSource.PREFILTER.value, 0) > 0


async def test_invalid_choices_and_judge_findings_fail_the_gate(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM(route=lambda request: "not json"))
    use_case = _use_case(container)
    trace = await run_session(
        use_case, make_pack(), ScriptedStudent(["karena lantai"]), name="s", mode=PlannerMode.HYBRID
    )
    report = session_report([trace], make_pack(), container.s3.config.guard)
    assert report["valid_choices"] == 0.0 and not report["gates"]["valid_choices"]
    finding = LeakFinding("s", 1, "Jawabannya gesekan?", "leak: names the answer")
    judged = session_report([trace], make_pack(), container.s3.config.guard, [finding])
    assert not judged["gates"]["leaks"] and not judged["passed"]


async def test_changed_approved_text_is_a_deterministic_finding(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM(route=_fake_prober))
    trace = await run_session(
        _use_case(container),
        make_pack(),
        ScriptedStudent(["gatau"]),
        name="s",
        mode=PlannerMode.TABLE,
    )
    first = trace.turns[0]
    assert first.result.probe is not None
    tampered = replace(
        first,
        result=replace(
            first.result, probe=replace(first.result.probe, question_text="Jawabannya gesekan?")
        ),
    )
    trace = replace(trace, turns=(tampered, *trace.turns[1:]))
    report = session_report([trace], make_pack(), container.s3.config.guard)
    assert report["leaks_deterministic"][0]["problem"] == "approved text changed"


async def test_classification_accuracy(settings: Settings) -> None:
    llm = ScriptedLLM(
        [classify_reply(), classify_reply(answer_type="correct_reasoned", misconceptions=[])]
    )
    items = [
        LabelledAnswer(
            "Kenapa berhenti?", "karena gayanya habis", AnswerType.MISCONCEPTION, (M_HABIS,)
        ),
        LabelledAnswer("Kenapa berhenti?", "ada yang menahan di lantai", AnswerType.MIXED, ()),
        LabelledAnswer("Kenapa berhenti?", "gatau", AnswerType.EVASIVE, ()),
    ]
    report = await run_classify(_container(settings, llm), make_pack(), items)
    assert report["type_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["misconception_accuracy"] == 1.0
    assert report["passed"] is False  # below 85%
    assert len(llm.requests) == 2  # "gatau" was prefiltered


async def test_persona_students_and_the_judge_use_eval_purposes(settings: Settings) -> None:
    llm = ScriptedLLM(
        [
            json.dumps({"answer": "karena dorongannya habis kak"}),
            json.dumps({"leak": False, "verdict": True, "reason": "praises the answer"}),
        ]
    )
    container = _container(settings, llm)
    ledger = UsageLedger("r", 1.0)
    (persona, *_) = load_personas()
    student = PersonaStudent(container.llm, persona, ledger)
    assert await student.answer("Kenapa?", ()) == "karena dorongannya habis kak"
    verdict = await LeakJudge(container.llm).judge(make_pack(), "Bagus! Kenapa?", "x", ledger)
    assert verdict.verdict and not verdict.leak
    assert [r.purpose for r in ledger.records] == [EvalPurpose.FAKE_STUDENT, EvalPurpose.LEAK_JUDGE]
    assert [r.model for r in ledger.records] == [settings.model_quality, settings.model_judge]


def test_loaders(tmp_path: Path) -> None:
    labelled = tmp_path / "labelled.json"
    labelled.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "question": "Kenapa?",
                        "answer": "habis",
                        "answer_type": "misconception",
                        "misconception_ids": [str(M_HABIS)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (item,) = load_labelled(labelled)
    assert item.misconception_ids == (M_HABIS,)
    assert len(load_personas()) == 7
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"scripts": [{"name": "x", "kind": "normal", "answers": []}]}))
    with pytest.raises(ValueError, match="at least one answer"):
        load_scripts(empty)


def test_percentile_and_accuracy() -> None:
    assert percentile([3.0, 1.0, 2.0, 4.0], 50) == 2.0
    assert percentile([3.0, 1.0, 2.0, 4.0], 95) == 4.0
    assert math.isnan(percentile([], 50))
    assert accuracy([]) == 0.0
    assert accuracy([(AnswerType.EVASIVE, AnswerType.EVASIVE)]) == 1.0


async def test_empty_classification_dataset_fails_the_gate(settings: Settings) -> None:
    llm = ScriptedLLM()
    report = await run_classify(_container(settings, llm), make_pack(), [])
    assert report["type_accuracy"] == 0.0
    assert report["passed"] is False
    assert llm.requests == []


async def test_failed_classifier_does_not_match_unsure_gold(settings: Settings) -> None:
    llm = ScriptedLLM([PermanentLLMError("provider unavailable")])
    report = await run_classify(
        _container(settings, llm),
        make_pack(),
        [LabelledAnswer("Kenapa berhenti?", "mungkin lantainya", AnswerType.UNSURE, ())],
    )
    assert report["type_accuracy"] == 0.0
    assert report["passed"] is False
    assert len(report["errors"]) == 1


def test_empty_sessions_fail_without_nan_latency(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM())
    report = session_report([], make_pack(), container.s3.config.guard)
    assert report["coverage"] == 0.0
    assert report["gates"]["coverage"] is False
    assert report["gates"]["valid_choices"] is False
    assert report["passed"] is False
    assert report["turn_latency_p50_s"] is None
    assert report["turn_latency_p95_s"] is None
    json.dumps(report, allow_nan=False)


async def test_unfinished_session_cannot_establish_coverage(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM(route=_fake_prober))
    trace = await run_session(
        _use_case(container),
        make_pack(),
        ScriptedStudent(["gatau"]),
        name="unfinished",
        mode=PlannerMode.TABLE,
        max_turns=1,
    )
    assert trace.final.action is TurnAction.PROBE
    report = session_report([trace], make_pack(), container.s3.config.guard)
    assert report["finished"] == 0
    assert report["coverage"] == 0.0
    assert report["gates"]["coverage"] is False
    assert report["passed"] is False


@pytest.mark.parametrize("mode", [PlannerMode.HYBRID, PlannerMode.TABLE])
async def test_provider_outage_cannot_establish_valid_choices(
    settings: Settings, mode: PlannerMode
) -> None:
    container = _container(
        settings, ScriptedLLM(route=lambda request: PermanentLLMError("provider unavailable"))
    )
    report = await run_sessions(
        container,
        make_pack(),
        scripted_students([StudentScript("outage", "normal", ("karena lantainya",))]),
        mode=mode,
        judge=False,
    )
    assert report["coverage"] == 1.0  # Approved fallbacks still protect session coverage.
    assert report["valid_choices"] == 0.0
    assert report["gates"]["valid_choices"] is False
    assert report["passed"] is False


@pytest.mark.parametrize("mode", [PlannerMode.HYBRID, PlannerMode.TABLE])
async def test_prefilter_only_sessions_have_valid_rule_choices(
    settings: Settings, mode: PlannerMode
) -> None:
    llm = ScriptedLLM()
    report = await run_sessions(
        _container(settings, llm),
        make_pack(),
        scripted_students([StudentScript("prefilter", "steering", ("gatau",))]),
        mode=mode,
        judge=False,
    )
    assert report["valid_choices"] == 1.0
    assert report["passed"] is True
    assert llm.requests == []


async def test_table_mode_needs_no_writer_for_valid_choices(settings: Settings) -> None:
    llm = ScriptedLLM(route=_fake_prober)
    report = await run_sessions(
        _container(settings, llm),
        make_pack(),
        scripted_students([StudentScript("table", "normal", ("karena lantainya",))]),
        mode=PlannerMode.TABLE,
        judge=False,
    )
    assert report["valid_choices"] == 1.0
    assert report["passed"] is True
    assert all(request.system.startswith("You label one answer") for request in llm.requests)


async def test_safety_pauses_do_not_supply_or_reduce_finished_coverage(settings: Settings) -> None:
    container = _container(settings, ScriptedLLM())
    paused = await run_session(
        _use_case(container),
        make_pack(),
        ScriptedStudent(["aku ingin mati"]),
        name="safety",
        mode=PlannerMode.TABLE,
    )
    assert paused.final.action is TurnAction.SAFETY_PAUSE
    paused_report = session_report([paused], make_pack(), container.s3.config.guard)
    assert paused_report["safety_pauses"] == 1
    assert paused_report["gates"]["coverage"] is False
    assert paused_report["passed"] is False

    finished = await run_session(
        _use_case(container),
        make_pack(),
        ScriptedStudent(["gatau"]),
        name="finished",
        mode=PlannerMode.TABLE,
    )
    mixed_report = session_report([paused, finished], make_pack(), container.s3.config.guard)
    assert mixed_report["finished"] == 1
    assert mixed_report["coverage"] == 1.0
    assert mixed_report["passed"] is True


async def test_transcripts_are_valid_s4_eval_input(settings: Settings, tmp_path: Path) -> None:
    container = _container(settings, ScriptedLLM(route=_fake_prober))
    report = await run_sessions(
        container,
        make_pack(),
        scripted_students([StudentScript("gatau", "steering", ("gatau",))]),
        mode=PlannerMode.TABLE,
        judge=False,
        pack_json=pack_payload(),
    )
    path = tmp_path / "transcripts.json"
    path.write_text(json.dumps(report["transcripts"]), encoding="utf-8")
    (case,) = load_cases(path)
    assert case.name == "gatau" and case.teacher_levels is None
    assert validate_session(case.session) == []
    assert len(case.session.turns) == 1 + make_pack().max_probes
