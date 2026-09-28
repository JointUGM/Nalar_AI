"""A small, valid Gaya dan Gerak context pack and history builders for S3 tests."""

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
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import (
    BankQuestion,
    ContextPack,
    MisconceptionRef,
    TargetConcept,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.session import HistoryTurn, TurnKind
from nalar_ai.subsystems.s3_socratic_prober.module import PROMPTS_DIR
from tests.support.s1 import MODELS

T_GESEK = UUID(int=0x11)  # friction
T_LEMBAM = UUID(int=0x12)  # inertia
M_HABIS = UUID(int=0x21)  # "the push runs out" (inertia)
M_KASAR = UUID(int=0x22)  # "only rough surfaces have friction"

ANCHOR = (
    "Kelereng didorong di lantai lalu dilepas. Kelereng itu makin pelan lalu berhenti. "
    "Kenapa kelereng itu berhenti?"
)
REFERENCE = (
    "Kelereng melambat lalu berhenti karena ada gaya gesek antara kelereng dan lantai yang "
    "arahnya melawan gerak. Tanpa gaya yang melawan, benda yang bergerak akan terus bergerak "
    "dengan kecepatan tetap karena kelembaman."
)

_QUESTIONS: dict[tuple[UUID, M], str] = {
    (
        T_GESEK,
        M.TRANSFER,
    ): "Bayangkan kelereng yang sama digelindingkan di atas es. Apa yang berbeda, dan kenapa?",
    (
        T_GESEK,
        M.REQUEST_JUSTIFICATION,
    ): "Apa yang membuatmu yakin? Apa saja yang menyentuh kelereng saat menggelinding?",
    (
        T_GESEK,
        M.COUNTER_EXAMPLE,
    ): "Lantai keramik itu licin dan halus, tapi kelereng tetap berhenti. Kenapa bisa begitu?",
    (
        T_GESEK,
        M.DECOMPOSE,
    ): "Kita pelan-pelan. Setelah tanganmu lepas, apa saja yang masih menyentuh kelereng?",
    (
        T_GESEK,
        M.REFUSE_AND_REDIRECT,
    ): "Aku tidak bisa memberi jawabannya, tapi aku penasaran pendapatmu. Kenapa kelereng itu akhirnya berhenti?",
    (
        T_GESEK,
        M.DEEPER_REASON,
    ): "Kalau lantainya diganti karpet, apa yang berubah pada gerak kelereng, dan kenapa?",
    (
        T_GESEK,
        M.EXPLAIN_MECHANISM,
    ): "Coba ceritakan apa yang terjadi pada kelereng dari lepas sampai berhenti?",
    (
        T_GESEK,
        M.SIMPLER_REASON,
    ): "Dengan kata-katamu sendiri, kenapa kelereng tidak menggelinding selamanya?",
    (
        T_LEMBAM,
        M.TRANSFER,
    ): "Sebuah bola ditendang di luar angkasa yang tidak ada udaranya. Apa yang terjadi pada bola itu, dan kenapa?",
    (
        T_LEMBAM,
        M.REQUEST_JUSTIFICATION,
    ): "Kenapa kamu berpikir begitu? Apa yang terjadi pada dorongan tanganmu setelah kelereng lepas?",
    (
        T_LEMBAM,
        M.COUNTER_EXAMPLE,
    ): "Kalau dorongannya habis, kenapa pesawat luar angkasa tetap melaju walau mesinnya mati?",
    (
        T_LEMBAM,
        M.DECOMPOSE,
    ): "Kita pecah jadi kecil. Saat kelereng sudah lepas, apakah tanganmu masih mendorongnya?",
    (
        T_LEMBAM,
        M.REFUSE_AND_REDIRECT,
    ): "Aku tidak bisa memberi tahu jawabannya. Menurutmu, apa yang membuat benda tetap bergerak setelah didorong?",
    (
        T_LEMBAM,
        M.DEEPER_REASON,
    ): "Kalau tidak ada yang menghambat sama sekali, apa yang terjadi pada kelereng, dan kenapa?",
    (
        T_LEMBAM,
        M.EXPLAIN_MECHANISM,
    ): "Coba jelaskan, apa yang membuat kelereng masih bergerak setelah lepas dari tanganmu?",
    (
        T_LEMBAM,
        M.SIMPLER_REASON,
    ): "Pakai contoh sehari-hari, kenapa benda yang sudah bergerak bisa terus bergerak?",
}

_SHORT = {T_GESEK: "gesek", T_LEMBAM: "lembam"}


def question_id(concept: UUID, move: M) -> str:
    return f"{_SHORT[concept]}-{move.value}"


def make_pack(**overrides: Any) -> ContextPack:
    bank = tuple(
        BankQuestion(
            id=question_id(concept, move),
            concept_id=concept,
            move=move,
            text=text,
            misconception_id=(
                {T_GESEK: M_KASAR, T_LEMBAM: M_HABIS}[concept]
                if move is M.COUNTER_EXAMPLE
                else None
            ),
        )
        for (concept, move), text in _QUESTIONS.items()
    )
    pack = ContextPack(
        anchor_problem=ANCHOR,
        reference_reasoning=REFERENCE,
        max_probes=5,
        max_duration_minutes=15,
        targets=(
            TargetConcept(T_GESEK, "Gaya gesek", "Gaya yang melawan gerak benda di permukaan."),
            TargetConcept(T_LEMBAM, "Kelembaman", "Benda bergerak tetap bergerak tanpa gaya."),
        ),
        misconceptions=(
            MisconceptionRef(
                M_HABIS,
                T_LEMBAM,
                "Benda berhenti karena gaya dorongnya habis.",
                ("gayanya habis", "dorongannya hilang", "tenaganya sudah tidak ada"),
            ),
            MisconceptionRef(
                M_KASAR,
                T_GESEK,
                "Gesekan hanya terjadi pada permukaan yang kasar.",
                ("lantainya licin jadi tidak ada gesekan", "cuma permukaan kasar yang menahan"),
            ),
        ),
        question_bank=bank,
        answer_terms=("gesekan", "gaya gesek", "kelembaman", "inersia", "hukum newton"),
    )
    return replace(pack, **overrides)


def question(concept: UUID, move: M) -> BankQuestion:
    return next(q for q in make_pack().question_bank if q.id == question_id(concept, move))


def pack_payload(pack: ContextPack | None = None) -> dict[str, Any]:
    """The pack as the backend sends it (ContextPackIn JSON)."""
    pack = pack or make_pack()
    return {
        "pack_version": 1,
        "anchor_problem": pack.anchor_problem,
        "reference_reasoning": pack.reference_reasoning,
        "max_probes": pack.max_probes,
        "max_duration_minutes": pack.max_duration_minutes,
        "targets": [
            {"id": str(t.id), "name": t.name, "description": t.description} for t in pack.targets
        ],
        "misconceptions": [
            {
                "id": str(m.id),
                "concept_id": str(m.concept_id),
                "statement": m.statement,
                "detection_cues": list(m.detection_cues),
            }
            for m in pack.misconceptions
        ],
        "question_bank": [
            {
                "id": q.id,
                "concept_id": str(q.concept_id),
                "move": q.move.value,
                "misconception_id": str(q.misconception_id) if q.misconception_id else None,
                "text": q.text,
            }
            for q in pack.question_bank
        ],
        "answer_terms": list(pack.answer_terms),
    }


def anchor(answer: str, *, misconceptions: tuple[UUID, ...] = ()) -> HistoryTurn:
    return HistoryTurn(
        turn_index=0,
        kind=TurnKind.ANCHOR,
        question_text=ANCHOR,
        answer_text=answer,
        misconception_ids=misconceptions,
    )


def probe(
    index: int,
    concept: UUID,
    move: M,
    answer: str,
    *,
    answer_type: AnswerType | None = None,
    misconceptions: tuple[UUID, ...] = (),
) -> HistoryTurn:
    asked = question(concept, move)
    return HistoryTurn(
        turn_index=index,
        kind=TurnKind.PROBE,
        question_text=asked.text,
        answer_text=answer,
        move=move,
        target_concept_id=concept,
        question_bank_id=asked.id,
        answer_type=answer_type,
        misconception_ids=misconceptions,
    )


def make_gateway(llm: ScriptedLLM) -> LLMGateway:
    return LLMGateway(
        port=llm, prompts=PromptRegistry.from_directories([PROMPTS_DIR]), models=MODELS
    )


def classify_reply(**changes: Any) -> str:
    """A classifier answer: by default, the student shows wrong idea m1 ("gaya habis")."""
    reply = {
        "answer_type": "misconception",
        "confidence": "high",
        "misconceptions": ["m1"],
        "frustration": False,
        "safety_concern": False,
        "key_phrase": "",
    }
    return json.dumps({**reply, **changes})


def choose_reply(**changes: Any) -> str:
    """A chooser answer: by default, the lembam counter-example (alias q11) lightly adapted."""
    reply = {
        "move": "counter_example",
        "question_id": "q11",
        "reason_code": "default",
        "reason": "",
        "question": "Kamu bilang gayanya habis. Kalau dorongannya habis, kenapa pesawat luar "
        "angkasa tetap melaju walau mesinnya mati?",
    }
    return json.dumps({**reply, **changes})


def by_prompt(*, classify: Iterable[Reply] = (), choose: Iterable[Reply] = ()) -> ScriptedLLM:
    """A ScriptedLLM that answers each S3 prompt from its own queue."""
    queues = {"classify": deque(classify), "choose": deque(choose)}

    def route(request: LLMRequest) -> Reply:
        name = "classify" if request.system.startswith("You label one answer") else "choose"
        queue = queues[name]
        return queue.popleft() if queue else PermanentLLMError(f"no {name} reply queued")

    return ScriptedLLM(route=route)


def history_payload(turns: Iterable[HistoryTurn]) -> list[dict[str, Any]]:
    """The history as the backend sends it (HistoryTurnIn JSON)."""
    return [
        {
            "turn_index": t.turn_index,
            "kind": t.kind.value,
            "question_text": t.question_text,
            "answer_text": t.answer_text,
            "move": t.move.value if t.move else None,
            "target_concept_id": str(t.target_concept_id) if t.target_concept_id else None,
            "question_bank_id": t.question_bank_id,
            "answer_type": t.answer_type.value if t.answer_type else None,
            "misconception_ids": [str(m) for m in t.misconception_ids],
        }
        for t in turns
    ]
