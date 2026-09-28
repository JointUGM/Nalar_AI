import asyncio
from dataclasses import replace

import pytest

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.ports import LLMRequest, LLMResponse, TransientLLMError
from nalar_ai.shared.enums import (
    AiPurpose,
    GuardResult,
    MoveReasonCode,
    MoveSource,
    PlannerMode,
)
from nalar_ai.shared.enums import ProbeStrategy as M
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s3_socratic_prober.application.next_turn import (
    NextTurnCommand,
    NextTurnUseCase,
    QuestionSource,
    TurnAction,
    TurnResult,
)
from nalar_ai.subsystems.s3_socratic_prober.application.policy import ProberPolicy
from nalar_ai.subsystems.s3_socratic_prober.domain.classification import ClassificationSource
from nalar_ai.subsystems.s3_socratic_prober.domain.controller import EndReason
from nalar_ai.subsystems.s3_socratic_prober.domain.guard import GuardThresholds
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.session import HistoryTurn
from nalar_ai.subsystems.s3_socratic_prober.infrastructure.config_loader import (
    load_prober_config,
)
from tests.support.s1 import make_embeddings
from tests.support.s3 import (
    M_HABIS,
    REFERENCE,
    T_GESEK,
    T_LEMBAM,
    anchor,
    by_prompt,
    choose_reply,
    classify_reply,
    make_gateway,
    make_pack,
    probe,
    question,
    question_id,
)

POLICY = ProberPolicy(
    turn_budget_s=4.5,
    classify_timeout_s=2.0,
    choose_timeout_s=2.5,
    embed_timeout_s=1.0,
    min_step_s=0.3,
    min_probes=4,
    transcript_turns=6,
    max_answer_chars=1500,
    guard=GuardThresholds(max_reference_similarity=0.8, min_approved_similarity=0.55),
)
CONFIG = load_prober_config()
FIRST = (anchor("kelerengnya berhenti karena gayanya habis"),)


async def _run(
    llm: ScriptedLLM,
    history: tuple[HistoryTurn, ...] = FIRST,
    *,
    mode: PlannerMode = PlannerMode.HYBRID,
    policy: ProberPolicy = POLICY,
    elapsed_seconds: int = 60,
) -> tuple[TurnResult, UsageLedger]:
    ledger = UsageLedger("req-s3", 1.0)
    use_case = NextTurnUseCase(
        llm=make_gateway(llm), embeddings=make_embeddings(), config=CONFIG, policy=policy
    )
    result = await use_case.execute(
        NextTurnCommand(make_pack(), history, mode, elapsed_seconds), ledger
    )
    return result, ledger


async def test_hybrid_turn_classifies_chooses_and_guards() -> None:
    result, ledger = await _run(by_prompt(classify=[classify_reply()], choose=[choose_reply()]))
    assert result.action is TurnAction.PROBE
    assert result.classification.answer_type is AnswerType.MISCONCEPTION
    assert result.classification.misconception_ids == (M_HABIS,)
    probe_ = result.probe
    assert probe_ is not None
    assert probe_.move is M.COUNTER_EXAMPLE
    assert probe_.allowed_moves == (M.COUNTER_EXAMPLE, M.EXPLAIN_MECHANISM, M.DECOMPOSE)
    assert probe_.target_concept_id == T_LEMBAM
    assert probe_.question_bank_id == question_id(T_LEMBAM, M.COUNTER_EXAMPLE)
    assert probe_.question_text.startswith("Kamu bilang gayanya habis.")
    assert probe_.question_source is QuestionSource.ADAPTED
    assert probe_.move_source is MoveSource.PLANNER
    assert probe_.reason_code is MoveReasonCode.DEFAULT
    assert probe_.reason == "Contoh pembanding, karena jawaban memuat ide keliru."
    assert probe_.guard_result is GuardResult.PASSED
    assert result.coverage == ((T_GESEK, False), (T_LEMBAM, True))
    assert result.warnings == ()
    assert [r.purpose for r in ledger.records] == [
        AiPurpose.TURN_ANALYZE,
        AiPurpose.PROBE_PLAN,
        AiPurpose.EMBEDDING,
    ]


async def test_the_writer_never_receives_the_reference() -> None:
    llm = by_prompt(classify=[classify_reply()], choose=[choose_reply()])
    await _run(llm)
    classify_request, choose_request = llm.requests
    assert REFERENCE in classify_request.blocks[0].text
    assert all(REFERENCE not in block.text for block in choose_request.blocks)


async def test_a_reasoned_departure_keeps_the_models_reason() -> None:
    reply = choose_reply(
        move="decompose",
        question_id="q12",
        reason_code="frustration",
        reason="Siswa terdengar bingung, jadi dipecah dulu.",
        question=question(T_LEMBAM, M.DECOMPOSE).text,
    )
    result, ledger = await _run(by_prompt(classify=[classify_reply()], choose=[reply]))
    assert result.probe is not None
    assert result.probe.move is M.DECOMPOSE
    assert result.probe.reason_code is MoveReasonCode.FRUSTRATION
    assert result.probe.reason == "Siswa terdengar bingung, jadi dipecah dulu."
    # the approved text came back unchanged, so no guard call was needed
    assert result.probe.question_source is QuestionSource.APPROVED
    assert result.probe.guard_result is GuardResult.NOT_RUN
    assert len(ledger.records) == 2


async def test_prefiltered_answers_cost_nothing() -> None:
    llm = ScriptedLLM()
    result, ledger = await _run(llm, (anchor("gatau"),))
    assert result.classification.source is ClassificationSource.PREFILTER
    assert result.probe is not None
    assert result.probe.move is M.DECOMPOSE
    assert result.probe.move_source is MoveSource.PREFILTER
    assert result.probe.question_text == question(T_GESEK, M.DECOMPOSE).text
    assert result.probe.guard_result is GuardResult.NOT_RUN
    assert llm.requests == [] and ledger.records == ()


async def test_distress_pauses_the_session_before_any_model_call() -> None:
    llm = ScriptedLLM()
    result, _ = await _run(llm, (anchor("aku pengen mati aja, capek"),))
    assert result.action is TurnAction.SAFETY_PAUSE
    assert result.safety_message == CONFIG.texts.safety_message
    assert result.classification.answer_type is AnswerType.SAFETY
    assert result.probe is None and llm.requests == []


async def test_the_classifier_can_raise_a_safety_concern() -> None:
    llm = by_prompt(classify=[classify_reply(safety_concern=True)])
    result, ledger = await _run(llm)
    assert result.action is TurnAction.SAFETY_PAUSE
    assert len(ledger.records) == 1


async def test_the_session_ends_at_the_turn_limit_after_classifying() -> None:
    history = (
        anchor("a"),
        *(probe(i, T_GESEK, M.DECOMPOSE, "karena lantainya") for i in range(1, 6)),
    )
    result, ledger = await _run(by_prompt(classify=[classify_reply()]), history)
    assert result.action is TurnAction.END
    assert result.end_reason is EndReason.TURN_LIMIT
    assert result.probe is None
    assert [r.purpose for r in ledger.records] == [AiPurpose.TURN_ANALYZE]


async def test_an_invalid_choice_falls_back_to_the_default_question() -> None:
    llm = by_prompt(classify=[classify_reply()], choose=[choose_reply(move="transfer")])
    result, _ = await _run(llm)
    assert result.probe is not None
    assert result.probe.move is M.COUNTER_EXAMPLE
    assert result.probe.move_source is MoveSource.FALLBACK_INVALID
    assert result.probe.question_text == question(T_LEMBAM, M.COUNTER_EXAMPLE).text
    assert "pilihan AI tidak lolos pemeriksaan" in result.probe.reason
    assert result.warnings == ("choice_fallback",)


async def test_provider_trouble_falls_back_at_every_step() -> None:
    llm = by_prompt(classify=[TransientLLMError("529")], choose=[TransientLLMError("529")])
    result, ledger = await _run(llm)
    assert result.classification.answer_type is AnswerType.UNSURE
    assert result.probe is not None
    assert result.probe.move is M.REQUEST_JUSTIFICATION
    assert result.probe.move_source is MoveSource.FALLBACK_ERROR
    assert result.warnings == ("classification_fallback", "choice_fallback")
    assert len(ledger.records) == 2  # both failed attempts are reported


async def test_a_blocked_adaptation_shows_the_chosen_question_verbatim() -> None:
    reply = choose_reply(question="Jawabanmu salah. Kenapa pesawat tetap melaju?")
    result, _ = await _run(by_prompt(classify=[classify_reply()], choose=[reply]))
    assert result.probe is not None
    assert result.probe.guard_result is GuardResult.BLOCKED_VERDICT
    assert result.probe.question_source is QuestionSource.APPROVED
    assert result.probe.question_text == question(T_LEMBAM, M.COUNTER_EXAMPLE).text
    assert result.probe.move_source is MoveSource.PLANNER


async def test_table_mode_shows_the_default_question_verbatim_without_a_writer_call() -> None:
    # Sonnet via SumoPod cannot write within the turn budget (DECISIONS P9), so table
    # mode, the latency fallback (design R-S3-1), makes no writer call at all.
    llm = by_prompt(classify=[classify_reply()], choose=[choose_reply()])
    result, ledger = await _run(llm, mode=PlannerMode.TABLE)
    assert result.probe is not None
    assert result.probe.allowed_moves == (M.COUNTER_EXAMPLE,)
    assert result.probe.move is M.COUNTER_EXAMPLE
    assert result.probe.move_source is MoveSource.DEFAULT
    assert result.probe.reason_code is MoveReasonCode.DEFAULT
    assert result.probe.question_text == question(T_LEMBAM, M.COUNTER_EXAMPLE).text
    assert result.probe.question_source is QuestionSource.APPROVED
    assert result.probe.guard_result is GuardResult.NOT_RUN
    assert [r.purpose for r in ledger.records] == [AiPurpose.TURN_ANALYZE]


class _SlowLLM:
    """Answers correctly, but only after `delay_s`."""

    def __init__(self, inner: ScriptedLLM, delay_s: float) -> None:
        self._inner, self._delay_s = inner, delay_s

    async def complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(self._delay_s)
        return await self._inner.complete(request)


@pytest.mark.parametrize(
    ("mode", "source"),
    [
        (PlannerMode.HYBRID, ClassificationSource.FALLBACK),
        (PlannerMode.TABLE, ClassificationSource.MODEL),
    ],
)
async def test_table_mode_gives_classify_the_whole_turn_budget(
    mode: PlannerMode, source: ClassificationSource
) -> None:
    policy = replace(POLICY, turn_budget_s=1.0, classify_timeout_s=0.1, min_step_s=0.05)
    llm = _SlowLLM(by_prompt(classify=[classify_reply()], choose=[choose_reply()]), 0.3)
    ledger = UsageLedger("req-s3", 1.0)
    use_case = NextTurnUseCase(
        llm=make_gateway(llm), embeddings=make_embeddings(), config=CONFIG, policy=policy
    )
    result = await use_case.execute(NextTurnCommand(make_pack(), FIRST, mode, 60), ledger)
    assert result.classification.source is source


async def test_a_spent_budget_skips_every_model_call() -> None:
    llm = ScriptedLLM()
    result, ledger = await _run(llm, policy=replace(POLICY, turn_budget_s=0.1))
    assert result.probe is not None
    assert result.probe.move_source is MoveSource.FALLBACK_ERROR
    assert llm.requests == [] and ledger.records == ()


async def test_coverage_forcing_explains_itself() -> None:
    history = (
        anchor("a"),
        probe(1, T_GESEK, M.REQUEST_JUSTIFICATION, "b"),
        probe(2, T_GESEK, M.DECOMPOSE, "c"),
        probe(3, T_GESEK, M.REFUSE_AND_REDIRECT, "kasih tau jawabannya dong"),
    )
    result, _ = await _run(ScriptedLLM(), history)  # "kasih tau jawabannya" is prefiltered
    assert result.probe is not None
    assert result.probe.move is M.TRANSFER
    assert result.probe.target_concept_id == T_GESEK
    assert result.probe.reason.startswith("Situasi baru: setiap konsep target wajib")


async def test_an_invalid_pack_or_history_is_rejected_before_any_call() -> None:
    llm = ScriptedLLM()
    use_case = NextTurnUseCase(
        llm=make_gateway(llm), embeddings=make_embeddings(), config=CONFIG, policy=POLICY
    )
    pack = replace(make_pack(), answer_terms=())
    with pytest.raises(InvalidInputError) as info:
        await use_case.execute(
            NextTurnCommand(pack, (), PlannerMode.HYBRID, 0), UsageLedger("r", 1.0)
        )
    assert info.value.details["problems"] == [
        "the pack has no answer_terms for the leak guard",
        "history is empty; it must start with the anchor turn",
    ]
    assert llm.requests == []
