"""S3 eval gates (AI-12, design doc s3 §16). Pure functions over session traces."""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from evals.s3.simulate import SessionTrace
from nalar_ai.shared.enums import MoveSource
from nalar_ai.subsystems.s3_socratic_prober.application.next_turn import (
    QuestionSource,
    TurnAction,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import ClassificationSource
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardLexicon, check_text
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack

GATES = {"coverage": 1.0, "valid_choices": 0.98, "classification_accuracy": 0.85}

_INVALID_SOURCES = frozenset({MoveSource.FALLBACK_INVALID, MoveSource.FALLBACK_ERROR})


@dataclass(frozen=True, slots=True)
class LeakFinding:
    session: str
    turn: int
    question: str
    problem: str


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; NaN for no values."""
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def accuracy(pairs: Sequence[tuple[AnswerType, AnswerType | None]]) -> float:
    """(expected, predicted) pairs; unavailable predictions and no evidence score zero."""
    if not pairs:
        return 0.0
    return sum(expected is predicted for expected, predicted in pairs) / len(pairs)


def deterministic_leaks(
    trace: SessionTrace, pack: ContextPack, lexicon: GuardLexicon
) -> list[LeakFinding]:
    """Re-run the text checks on everything shown, and check verbatim texts are verbatim."""
    bank = {q.id: q.text for q in pack.question_bank}
    findings: list[LeakFinding] = []
    for index, turn in enumerate(trace.turns):
        probe = turn.result.probe
        if probe is None:
            continue
        approved = bank[probe.question_bank_id]
        if probe.question_source is QuestionSource.APPROVED:
            if probe.question_text != approved:
                findings.append(
                    LeakFinding(trace.name, index, probe.question_text, "approved text changed")
                )
            continue
        answers = [t.answer for t in trace.turns[: index + 1]]
        blocked = check_text(probe.question_text, approved, answers, pack.answer_terms, lexicon)
        if blocked is not None:
            findings.append(LeakFinding(trace.name, index, probe.question_text, blocked.value))
    return findings


def session_report(
    traces: Sequence[SessionTrace],
    pack: ContextPack,
    lexicon: GuardLexicon,
    judge_findings: Sequence[LeakFinding] = (),
) -> dict[str, Any]:
    probes = [turn.result.probe for trace in traces for turn in trace.turns if turn.result.probe]
    # Prefilter choices need no model evidence. Failed attempts must stay in the denominator.
    consulted = [
        turn.result
        for trace in traces
        for turn in trace.turns
        if turn.result.probe and turn.result.probe.move_source is not MoveSource.PREFILTER
    ]
    invalid = [
        result
        for result in consulted
        if result.probe
        and (
            result.probe.move_source in _INVALID_SOURCES
            or (
                result.probe.move_source is MoveSource.DEFAULT
                and result.classification.source is ClassificationSource.FALLBACK
            )
        )
    ]
    valid_rate = 1 - len(invalid) / len(consulted) if consulted else float(bool(probes))
    finished = [t for t in traces if t.final.action is TurnAction.END]
    covered = [t for t in finished if all(challenged for _, challenged in t.final.coverage)]
    coverage_rate = len(covered) / len(finished) if finished else 0.0
    deterministic = [f for t in traces for f in deterministic_leaks(t, pack, lexicon)]
    latencies = [turn.latency_s for trace in traces for turn in trace.turns]
    costs = [sum(turn.cost_usd for turn in trace.turns) for trace in traces]
    gates = {
        "leaks": not deterministic and not judge_findings,
        "coverage": coverage_rate >= GATES["coverage"],
        "valid_choices": valid_rate >= GATES["valid_choices"],
    }
    return {
        "sessions": len(traces),
        "finished": len(finished),
        "safety_pauses": sum(t.final.action is TurnAction.SAFETY_PAUSE for t in traces),
        "coverage": round(coverage_rate, 4),
        "valid_choices": round(valid_rate, 4),
        "leaks_deterministic": [_finding(f) for f in deterministic],
        "leaks_judge": [_finding(f) for f in judge_findings],
        "turn_latency_p50_s": round(percentile(latencies, 50), 3) if latencies else None,
        "turn_latency_p95_s": round(percentile(latencies, 95), 3) if latencies else None,
        "cost_per_session_usd": round(sum(costs) / len(costs), 6) if costs else 0.0,
        "move_sources": dict(Counter(p.move_source.value for p in probes)),
        "guard_results": dict(Counter(p.guard_result.value for p in probes)),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _finding(finding: LeakFinding) -> dict[str, Any]:
    return {
        "session": finding.session,
        "turn": finding.turn,
        "question": finding.question,
        "problem": finding.problem,
    }
