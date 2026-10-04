"""S3 eval gate (AI-12). Real runs cost money and need a configured .env.

    uv run python -m evals.s3.run classify --data evals/s3/datasets/labelled.json
    uv run python -m evals.s3.run scripts --scripts evals/s3/datasets/adversarial.json --judge
    uv run python -m evals.s3.run personas --runs 10 --mode table --judge
    uv run python -m evals.s3.run personas --runs 3 --transcripts evals/s4/datasets/personas.json

--pack defaults to evals/s3/datasets/sample_pack.json; use the pedagogy lead's demo pack.
Exit code 1 when a gate fails.
"""

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.s3.data import (
    SAMPLE_PACK,
    LabelledAnswer,
    StudentScript,
    load_labelled,
    load_pack,
    load_personas,
    load_scripts,
)
from evals.s3.metrics import GATES, LeakFinding, accuracy, session_report
from evals.s3.simulate import (
    EVAL_PROMPTS_DIR,
    LeakJudge,
    PersonaStudent,
    ScriptedStudent,
    SessionTrace,
    Student,
    run_session,
)
from nalar_ai.container import Container, build_container
from nalar_ai.settings import get_settings
from nalar_ai.shared.enums import PlannerMode
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.classify_answer import (
    ClassifyAnswerUseCase,
)
from nalar_ai.subsystems.s3_socratic_prober.application.next_turn import NextTurnUseCase
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import (
    Classification,
    ClassificationSource,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack
from nalar_ai.subsystems.s3_socratic_prober.domain.prefilter import prefilter
from nalar_ai.subsystems.s3_socratic_prober.domain.session import (
    HistoryTurn,
    SessionView,
    TurnKind,
)

CLASSIFY_TIMEOUT_S = 10.0  # accuracy, not latency, is measured here


async def run_classify(
    container: Container, pack: ContextPack, items: Sequence[LabelledAnswer]
) -> dict[str, Any]:
    policy = container.s3.policy
    classify = ClassifyAnswerUseCase(
        llm=container.llm,
        transcript_turns=policy.transcript_turns,
        max_answer_chars=policy.max_answer_chars,
    )
    rows: list[dict[str, Any]] = []
    pairs: list[tuple[AnswerType, AnswerType | None]] = []
    for number, item in enumerate(items):
        view = SessionView((HistoryTurn(0, TurnKind.ANCHOR, item.question, item.answer),))
        hit = prefilter(item.answer, container.s3.config.prefilter)
        result = (
            Classification.from_prefilter(hit)
            if hit is not None
            else await classify.execute(
                pack,
                view,
                UsageLedger(f"eval-classify-{number}", 1.0),
                timeout_s=CLASSIFY_TIMEOUT_S,
            )
        )
        pairs.append(
            (
                item.answer_type,
                result.answer_type if result.source is not ClassificationSource.FALLBACK else None,
            )
        )
        rows.append(
            {
                "answer": item.answer,
                "expected": item.answer_type.value,
                "predicted": result.answer_type.value,
                "source": result.source.value,
                "misconception_ok": (
                    not item.misconception_ids
                    or result.primary_misconception in item.misconception_ids
                ),
            }
        )
    type_accuracy = accuracy(pairs)
    with_misconceptions = [r for i, r in zip(items, rows, strict=True) if i.misconception_ids]
    passed = type_accuracy >= GATES["classification_accuracy"]
    return {
        "items": len(items),
        "type_accuracy": round(type_accuracy, 4),
        "misconception_accuracy": (
            round(
                sum(r["misconception_ok"] for r in with_misconceptions) / len(with_misconceptions),
                4,
            )
            if with_misconceptions
            else None
        ),
        "errors": [
            r
            for r in rows
            if r["expected"] != r["predicted"] or r["source"] == ClassificationSource.FALLBACK.value
        ],
        "gates": {"classification_accuracy": passed},
        "passed": passed,
    }


async def run_sessions(
    container: Container,
    pack: ContextPack,
    students: Sequence[tuple[str, Student]],
    *,
    mode: PlannerMode,
    judge: bool,
    pack_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """pack_json: also return the sessions as S4 eval input under "transcripts"."""
    use_case = NextTurnUseCase(
        llm=container.llm,
        embeddings=container.embeddings,
        config=container.s3.config,
        policy=container.s3.policy,
    )
    traces = [
        await run_session(use_case, pack, student, name=name, mode=mode)
        for name, student in students
    ]
    findings = await _judge(container, pack, traces) if judge else []
    report = session_report(traces, pack, container.s3.config.guard, findings)
    report["mode"] = mode.value
    report["judged"] = judge
    if pack_json is not None:
        report["transcripts"] = transcripts_payload(traces, pack_json)
    return report


def transcripts_payload(
    traces: Sequence[SessionTrace], pack_json: dict[str, Any]
) -> dict[str, Any]:
    """Finished sessions as S4 eval input. Turn ids are stable per session name and index."""
    return {
        "context_pack": pack_json,
        "sessions": [
            {
                "name": trace.name,
                "turns": [
                    {
                        "turn_id": str(
                            uuid.uuid5(uuid.NAMESPACE_URL, f"{trace.name}/{t.turn_index}")
                        ),
                        "turn_index": t.turn_index,
                        "kind": t.kind.value,
                        "question_text": t.question_text,
                        "answer_text": t.answer_text,
                        "move": t.move.value if t.move else None,
                        "target_concept_id": (
                            str(t.target_concept_id) if t.target_concept_id else None
                        ),
                    }
                    for t in trace.history
                ],
            }
            for trace in traces
        ],
    }


async def _judge(
    container: Container, pack: ContextPack, traces: Sequence[SessionTrace]
) -> list[LeakFinding]:
    judge = LeakJudge(container.llm)
    findings: list[LeakFinding] = []
    for trace in traces:
        for index, turn in enumerate(trace.turns):
            probe = turn.result.probe
            if probe is None:
                continue
            verdict = await judge.judge(
                pack, probe.question_text, turn.answer, UsageLedger(f"judge-{trace.name}", 1.0)
            )
            if verdict.leak or verdict.verdict:
                problem = ("leak" if verdict.leak else "verdict") + f": {verdict.reason}"
                findings.append(LeakFinding(trace.name, index, probe.question_text, problem))
    return findings


def scripted_students(scripts: Sequence[StudentScript]) -> list[tuple[str, Student]]:
    return [(script.name, ScriptedStudent(script.answers)) for script in scripts]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="S3 eval gate (AI-12)")
    parser.add_argument("--pack", type=Path, default=SAMPLE_PACK)
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    commands = parser.add_subparsers(dest="command", required=True)
    classify = commands.add_parser("classify", help="answer-type accuracy (gate: 85%%)")
    classify.add_argument("--data", type=Path, required=True)
    for name, help_text in (
        ("scripts", "scripted adversarial/steering students"),
        ("personas", "Sonnet fake students"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--mode", choices=[m.value for m in PlannerMode], default="hybrid")
        command.add_argument("--judge", action="store_true", help="Opus leak judge (costs more)")
        command.add_argument("--transcripts", type=Path, help="save sessions as S4 eval input")
        if name == "scripts":
            command.add_argument("--scripts", type=Path, required=True)
        else:
            command.add_argument("--runs", type=int, default=1, help="runs per persona")
    args = parser.parse_args(argv)
    report = asyncio.run(_run(args))
    transcripts = report.pop("transcripts", None)
    if transcripts is not None:
        dump = json.dumps(transcripts, indent=2, ensure_ascii=False)
        args.transcripts.write_text(dump + "\n", encoding="utf-8")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("PASS" if report["passed"] else "FAIL", file=sys.stderr)
    return 0 if report["passed"] else 1


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    container = build_container(get_settings(), extra_prompt_directories=(EVAL_PROMPTS_DIR,))
    try:
        pack = load_pack(args.pack)
        if args.command == "classify":
            return await run_classify(container, pack, load_labelled(args.data))
        mode = PlannerMode(args.mode)
        if args.command == "scripts":
            students = scripted_students(load_scripts(args.scripts))
        else:
            ledger = UsageLedger("eval-students", 50.0)
            students = [
                (f"{persona.name}-{run}", PersonaStudent(container.llm, persona, ledger))
                for persona in load_personas()
                for run in range(args.runs)
            ]
        return await run_sessions(
            container,
            pack,
            students,
            mode=mode,
            judge=args.judge,
            pack_json=(
                json.loads(args.pack.read_text(encoding="utf-8")) if args.transcripts else None
            ),
        )
    finally:
        await container.aclose()


if __name__ == "__main__":
    raise SystemExit(main())
