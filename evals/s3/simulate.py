"""Whole sessions through the real S3 pipeline, with the backend's part played by this loop."""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from evals.s3.data import Persona
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.enums import PlannerMode
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s3_socratic_prober.application.next_turn import (
    NextTurnCommand,
    NextTurnUseCase,
    TurnResult,
)
from nalar_ai.subsystems.s3_socratic_prober.application.rendering import transcript
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack
from nalar_ai.subsystems.s3_socratic_prober.domain.session import HistoryTurn, TurnKind

EVAL_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class Student(Protocol):
    async def answer(self, question: str, history: Sequence[HistoryTurn]) -> str: ...


class ScriptedStudent:
    def __init__(self, answers: Sequence[str]) -> None:
        self._answers = list(answers)
        self._next = 0

    async def answer(self, question: str, history: Sequence[HistoryTurn]) -> str:
        answer = self._answers[min(self._next, len(self._answers) - 1)]
        self._next += 1
        return answer


class _StudentOut(BaseModel):
    answer: str


class PersonaStudent:
    """Sonnet plays a student type (AI design §15). Its calls go through the gateway too."""

    PROMPT_ID = "eval.fake_student"

    def __init__(self, llm: LLMGateway, persona: Persona, ledger: UsageLedger) -> None:
        self._llm = llm
        self._persona = persona
        self._ledger = ledger

    async def answer(self, question: str, history: Sequence[HistoryTurn]) -> str:
        out = await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "persona": fence_untrusted("persona", self._persona.description),
                "transcript": transcript(history, max_answer_chars=1500),
                "question": fence_untrusted("question", question),
            },
            output_model=_StudentOut,
            ledger=self._ledger,
        )
        return out.answer


class JudgeOut(BaseModel):
    leak: bool
    verdict: bool
    reason: str


class LeakJudge:
    """Opus reads the hidden reference and one shown question (AI-12 gate)."""

    PROMPT_ID = "eval.leak_judge"

    def __init__(self, llm: LLMGateway) -> None:
        self._llm = llm

    async def judge(
        self, pack: ContextPack, question: str, previous_answer: str, ledger: UsageLedger
    ) -> JudgeOut:
        return await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "reference": fence_untrusted("reference", pack.reference_reasoning),
                "answer_terms": ", ".join(pack.answer_terms),
                "previous_answer": fence_untrusted("student_answer", previous_answer),
                "question": fence_untrusted("question", question),
            },
            output_model=JudgeOut,
            ledger=ledger,
        )


@dataclass(frozen=True, slots=True)
class TurnTrace:
    answer: str  # the answer this decision was about
    result: TurnResult
    latency_s: float
    cost_usd: float


@dataclass(frozen=True, slots=True)
class SessionTrace:
    name: str
    mode: PlannerMode
    turns: tuple[TurnTrace, ...]
    history: tuple[HistoryTurn, ...]

    @property
    def final(self) -> TurnResult:
        return self.turns[-1].result


async def run_session(
    use_case: NextTurnUseCase,
    pack: ContextPack,
    student: Student,
    *,
    name: str,
    mode: PlannerMode,
    seconds_per_turn: int = 90,
    max_turns: int = 15,
    budget_usd: float = 1.0,
    clock: Callable[[], float] = time.perf_counter,
) -> SessionTrace:
    first = await student.answer(pack.anchor_problem, ())
    history = [HistoryTurn(0, TurnKind.ANCHOR, pack.anchor_problem, first)]
    traces: list[TurnTrace] = []
    for step in range(max_turns):
        ledger = UsageLedger(f"eval-{name}-{step}", budget_usd)
        started = clock()
        result = await use_case.execute(
            NextTurnCommand(pack, tuple(history), mode, step * seconds_per_turn), ledger
        )
        traces.append(
            TurnTrace(history[-1].answer_text, result, clock() - started, ledger.total_cost_usd)
        )
        analysis = result.classification
        history[-1] = replace(
            history[-1],
            answer_type=analysis.answer_type,
            misconception_ids=analysis.misconception_ids,
        )
        probe = result.probe
        if probe is None:
            break
        answer = await student.answer(probe.question_text, tuple(history))
        history.append(
            HistoryTurn(
                turn_index=len(history),
                kind=TurnKind.PROBE,
                question_text=probe.question_text,
                answer_text=answer,
                move=probe.move,
                target_concept_id=probe.target_concept_id,
                question_bank_id=probe.question_bank_id,
            )
        )
    return SessionTrace(name, mode, tuple(traces), tuple(history))
