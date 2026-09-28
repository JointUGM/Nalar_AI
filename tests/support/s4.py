"""A finished Gaya dan Gerak session (the S3 test pack) and scripted S4 model replies."""

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any
from uuid import UUID

from nalar_ai.platform.llm.fakes import Reply, ScriptedLLM
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.llm.ports import LLMRequest, PermanentLLMError
from nalar_ai.platform.prompts.registry import PromptRegistry
from nalar_ai.shared.enums import ConceptOutcome, RubricDimension
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s4_session_evaluator.application.policy import EvaluatorPolicy
from nalar_ai.subsystems.s4_session_evaluator.domain.session import (
    EvalMisconception,
    EvalPack,
    EvalSession,
    EvalTarget,
    EvalTurn,
    Rubric,
    TurnKind,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import (
    DraftConcept,
    DraftEvidence,
    DraftScore,
    DraftTurnQuality,
    Evaluation,
    ScoringDraft,
    VerificationLimits,
    verify,
)
from nalar_ai.subsystems.s4_session_evaluator.module import PROMPTS_DIR
from tests.support.s1 import MODELS
from tests.support.s3 import ANCHOR, M_HABIS, M_KASAR, REFERENCE, T_GESEK, T_LEMBAM, question

RUBRIC_TEXT: dict[str, list[str]] = {
    "claim": [
        "Tidak ada klaim.",
        "Klaim tidak jelas.",
        "Klaim sebagian benar.",
        "Klaim benar.",
        "Klaim benar dan tepat sasaran.",
    ],
    "evidence": [
        "Tidak ada bukti.",
        "Bukti tidak relevan.",
        "Satu bukti relevan.",
        "Beberapa bukti relevan.",
        "Bukti relevan dan dikaitkan dengan klaim.",
    ],
    "mechanism": [
        "Tidak ada mekanisme.",
        "Mekanisme keliru.",
        "Mekanisme sebagian.",
        "Mekanisme benar.",
        "Mekanisme benar dan runtut.",
    ],
    "transfer": [
        "Tidak mencoba situasi baru.",
        "Situasi baru dijawab keliru.",
        "Situasi baru dijawab sebagian.",
        "Situasi baru dijawab benar.",
        "Situasi baru dijawab benar dengan alasan.",
    ],
}

TURN_IDS = tuple(UUID(int=0x100 + i) for i in range(5))

ANSWERS = (
    "Kelerengnya berhenti karena gayanya habis.",
    "Hmm, mungkin karena di luar angkasa tidak ada yang menghambat, jadi tetap jalan.",
    "Lantainya menahan kelereng, ada gesekan jadi pelan-pelan berhenti.",
    "Di es lebih jauh karena licin, gesekannya kecil.",
    "gatau",
)

_PROBES = (
    (T_LEMBAM, M.COUNTER_EXAMPLE),
    (T_GESEK, M.REQUEST_JUSTIFICATION),
    (T_GESEK, M.TRANSFER),
    (T_LEMBAM, M.TRANSFER),
)


def make_pack() -> EvalPack:
    return EvalPack(
        anchor_problem=ANCHOR,
        reference_reasoning=REFERENCE,
        targets=(
            EvalTarget(T_GESEK, "Gaya gesek", "Gaya yang melawan gerak benda di permukaan."),
            EvalTarget(T_LEMBAM, "Kelembaman", "Benda bergerak tetap bergerak tanpa gaya."),
        ),
        misconceptions=(
            EvalMisconception(
                M_HABIS,
                T_LEMBAM,
                "Benda berhenti karena gaya dorongnya habis.",
                ("gayanya habis", "dorongannya hilang"),
            ),
            EvalMisconception(
                M_KASAR,
                T_GESEK,
                "Gesekan hanya terjadi pada permukaan yang kasar.",
                ("lantainya licin jadi tidak ada gesekan",),
            ),
        ),
        answer_terms=("gesekan", "gaya gesek", "kelembaman", "inersia", "hukum newton"),
    )


def make_rubric() -> Rubric:
    return Rubric({RubricDimension(key): tuple(value) for key, value in RUBRIC_TEXT.items()})


def make_turns(answers: Iterable[str] = ANSWERS) -> tuple[EvalTurn, ...]:
    turns: list[EvalTurn] = []
    for index, answer in enumerate(answers):
        if index == 0:
            turns.append(EvalTurn(TURN_IDS[0], 0, TurnKind.ANCHOR, ANCHOR, answer))
            continue
        concept, move = _PROBES[index - 1]
        turns.append(
            EvalTurn(
                TURN_IDS[index],
                index,
                TurnKind.PROBE,
                question(concept, move).text,
                answer,
                move=move,
                target_concept_id=concept,
            )
        )
    return tuple(turns)


def make_session(**overrides: Any) -> EvalSession:
    session = EvalSession(pack=make_pack(), rubric=make_rubric(), turns=make_turns())
    return replace(session, **overrides)


def scoring_payload(**changes: Any) -> dict[str, Any]:
    """A valid scoring answer for make_session(): every quote occurs in the cited answer."""
    reply: dict[str, Any] = {
        "scores": [
            {
                "dimension": "claim",
                "evidence": [{"turn": "t2", "quote": "ada gesekan jadi pelan-pelan berhenti"}],
                "rationale": "Klaim tentang gesekan muncul setelah pertanyaan kedua.",
                "level": 3,
            },
            {
                "dimension": "evidence",
                "evidence": [{"turn": "t2", "quote": "Lantainya menahan kelereng"}],
                "rationale": "Menyebut lantai yang menahan kelereng sebagai bukti.",
                "level": 2,
            },
            {
                "dimension": "mechanism",
                "evidence": [
                    {"turn": "t1", "quote": "tidak ada yang menghambat, jadi tetap jalan"}
                ],
                "rationale": "Mulai menjelaskan bahwa tanpa hambatan benda tetap bergerak.",
                "level": 2,
            },
            {
                "dimension": "transfer",
                "evidence": [{"turn": "t3", "quote": "Di es lebih jauh karena licin"}],
                "rationale": "Menerapkan ide ke situasi es dengan alasan.",
                "level": 3,
            },
        ],
        "concepts": [
            {
                "concept": "c1",
                "evidence_turn": "t3",
                "initial_misconception": None,
                "outcome": "mastered",
                "misconception": None,
            },
            {
                "concept": "c2",
                "evidence_turn": "t1",
                "initial_misconception": "m1",
                "outcome": "developing",
                "misconception": None,
            },
        ],
        "turn_quality": [
            {"turn": "t0", "quality": 1},
            {"turn": "t1", "quality": 2},
            {"turn": "t2", "quality": 3},
            {"turn": "t3", "quality": 3},
            {"turn": "t4", "quality": 0},
        ],
        "summary": "Siswa mengubah pendapat tentang gaya yang habis dan mulai memakai gesekan.",
    }
    return {**reply, **changes}


def scoring_reply(**changes: Any) -> str:
    return json.dumps(scoring_payload(**changes))


def draft(payload: dict[str, Any] | None = None) -> ScoringDraft:
    """A scoring payload as the domain sees it (the application converts the model's JSON)."""
    p = scoring_payload() if payload is None else payload
    return ScoringDraft(
        scores=tuple(
            DraftScore(
                dimension=RubricDimension(s["dimension"]),
                evidence=tuple(DraftEvidence(e["turn"], e["quote"]) for e in s["evidence"]),
                rationale=s["rationale"],
                level=s["level"],
            )
            for s in p["scores"]
        ),
        concepts=tuple(
            DraftConcept(
                concept=c["concept"],
                evidence_turn=c["evidence_turn"],
                initial_misconception=c["initial_misconception"],
                outcome=ConceptOutcome(c["outcome"]),
                misconception=c["misconception"],
            )
            for c in p["concepts"]
        ),
        turn_quality=tuple(DraftTurnQuality(q["turn"], q["quality"]) for q in p["turn_quality"]),
        summary=p["summary"],
    )


def make_evaluation() -> Evaluation:
    evaluation, errors = verify(draft(), make_session(), VerificationLimits())
    assert evaluation is not None, errors
    return evaluation


def reflection_reply(**changes: Any) -> str:
    reply = {
        "strengths": "Kamu memakai contoh lantai dan es untuk menjelaskan pikiranmu.",
        "changed_mind": "Awalnya kamu bilang gayanya habis, lalu kamu memikirkan lagi apa yang "
        "menghambat kelereng.",
        "question": "Kalau kelereng digelindingkan di karpet, apa yang akan kamu perhatikan?",
    }
    return json.dumps({**reply, **changes})


def make_gateway(llm: ScriptedLLM) -> LLMGateway:
    return LLMGateway(
        port=llm,
        prompts=PromptRegistry.from_directories([PROMPTS_DIR]),
        models=MODELS,
        base_delay_s=0.0,
    )


def make_policy() -> EvaluatorPolicy:
    return EvaluatorPolicy(
        score_timeout_s=5.0,
        reflect_timeout_s=5.0,
        max_attempts=2,
        max_answer_chars=1500,
        limits=VerificationLimits(),
    )


def by_prompt(*, score: Iterable[Reply] = (), reflect: Iterable[Reply] = ()) -> ScriptedLLM:
    """A ScriptedLLM that answers each S4 prompt from its own queue."""
    queues = {"score": deque(score), "reflect": deque(reflect)}

    def route(request: LLMRequest) -> Reply:
        name = "score" if request.system.startswith("You score") else "reflect"
        queue = queues[name]
        return queue.popleft() if queue else PermanentLLMError(f"no {name} reply queued")

    return ScriptedLLM(route=route)


def rubric_payload() -> dict[str, list[str]]:
    return {key: list(value) for key, value in RUBRIC_TEXT.items()}


def turns_payload(turns: Iterable[EvalTurn] | None = None) -> list[dict[str, Any]]:
    """The turns as the backend sends them (EvalTurnIn JSON)."""
    return [
        {
            "turn_id": str(t.turn_id),
            "turn_index": t.turn_index,
            "kind": t.kind.value,
            "question_text": t.question_text,
            "answer_text": t.answer_text,
            "move": t.move.value if t.move else None,
            "target_concept_id": str(t.target_concept_id) if t.target_concept_id else None,
        }
        for t in (make_turns() if turns is None else turns)
    ]
