import json
import uuid

import pytest

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.shared.enums import AiPurpose, RetrievalSource
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.dedupe_concepts import (
    DedupeConceptsUseCase,
    DedupItem,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.dedup_policy import (
    Candidate,
    DedupBasis,
    DedupThresholds,
)
from tests.support.s1 import make_gateway

GREY = Candidate(uuid.uuid4(), "Gesekan", "Gaya yang menghambat gerak.", 0.86)
CLOSE = Candidate(uuid.uuid4(), "Gaya gesek", "Gaya yang melawan gerak.", 0.95)


def _use_case(llm: ScriptedLLM) -> DedupeConceptsUseCase:
    return DedupeConceptsUseCase(llm=make_gateway(llm), thresholds=DedupThresholds())


async def test_clear_cases_need_no_model_call() -> None:
    llm = ScriptedLLM()
    decisions = await _use_case(llm).execute(
        [
            DedupItem("n1", "Gaya gesek kinetis", "desc", (CLOSE,)),
            DedupItem("n2", "Massa", "desc", ()),
        ],
        UsageLedger("r", 1.0),
    )
    assert [(d.key, d.action, d.basis) for d in decisions] == [
        ("n1", "link", DedupBasis.SIMILARITY),
        ("n2", "create", DedupBasis.NO_CANDIDATE),
    ]
    assert llm.requests == []


async def test_grey_zone_is_judged_and_traced() -> None:
    llm = ScriptedLLM([json.dumps({"same_as": "e1", "reason": "same idea"})])
    ledger = UsageLedger("r", 1.0)
    (decision,) = await _use_case(llm).execute(
        [DedupItem("n1", "Gaya gesek", "desc", (GREY,))], ledger
    )
    assert (decision.action, decision.existing_concept_id, decision.basis) == (
        "link",
        GREY.concept_id,
        DedupBasis.JUDGE,
    )
    assert decision.similarity == 0.86 and decision.reason == "same idea"
    (record,) = ledger.records
    assert record.purpose is AiPurpose.KB_DEDUP and record.model == "claude-haiku-4-5"
    assert [(r.source, r.id, r.score) for r in record.retrieval] == [
        (RetrievalSource.CONCEPT, GREY.concept_id, 0.86)
    ]
    assert llm.requests[0].effort is None


async def test_judge_saying_null_creates() -> None:
    llm = ScriptedLLM([json.dumps({"same_as": None, "reason": "narrower idea"})])
    (decision,) = await _use_case(llm).execute(
        [DedupItem("n1", "Gaya gesek", "d", (GREY,))], UsageLedger("r", 1.0)
    )
    assert (decision.action, decision.basis, decision.existing_concept_id) == (
        "create",
        DedupBasis.JUDGE,
        None,
    )


async def test_unknown_judge_alias_is_repaired() -> None:
    llm = ScriptedLLM(
        [json.dumps({"same_as": "e7", "reason": "x"}), json.dumps({"same_as": "e1", "reason": "y"})]
    )
    (decision,) = await _use_case(llm).execute(
        [DedupItem("n1", "Gaya gesek", "d", (GREY,))], UsageLedger("r", 1.0)
    )
    assert decision.action == "link"
    assert "same_as must be one of e1 or null" in llm.requests[1].blocks[-1].text


async def test_duplicate_keys_are_rejected() -> None:
    with pytest.raises(InvalidInputError):
        await _use_case(ScriptedLLM()).execute(
            [DedupItem("n1", "a", "", ()), DedupItem("n1", "b", "", ())], UsageLedger("r", 1.0)
        )
