import json
import uuid

from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.shared.enums import AiPurpose, RetrievalPath, RetrievalSource
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.align_cp import (
    AlignCpUseCase,
    AlignItem,
    CpBasis,
    CpCandidate,
)
from tests.support.s1 import make_gateway

S1 = CpCandidate(
    uuid.uuid4(), "Pemahaman IPA", "Mengidentifikasi gaya dan pengaruhnya pada gerak.", 0.62
)
S2 = CpCandidate(uuid.uuid4(), "Pemahaman IPA", "Menjelaskan tekanan pada zat.", 0.41)


def _use_case(llm: ScriptedLLM) -> AlignCpUseCase:
    return AlignCpUseCase(llm=make_gateway(llm), min_similarity=0.25)


async def test_picks_a_statement_and_traces_candidates() -> None:
    llm = ScriptedLLM([json.dumps({"choice": "s1", "reason": "about force and motion"})])
    ledger = UsageLedger("r", 1.0)
    (alignment,) = await _use_case(llm).execute(
        [AlignItem("n1", "Gaya gesek", "Melawan gerak.", (S2, S1))], ledger
    )
    assert (alignment.outcome_id, alignment.basis) == (S1.outcome_id, CpBasis.JUDGE)
    (record,) = ledger.records
    assert record.purpose is AiPurpose.CP_ALIGN
    assert [(r.source, r.path, r.rank) for r in record.retrieval] == [
        (RetrievalSource.CP_OUTCOME, RetrievalPath.SEARCH, 1),
        (RetrievalSource.CP_OUTCOME, RetrievalPath.SEARCH, 2),
    ]
    assert record.retrieval[0].id == S1.outcome_id  # ranked by similarity


async def test_none_is_an_honest_answer() -> None:
    llm = ScriptedLLM([json.dumps({"choice": "none", "reason": "no statement fits"})])
    (alignment,) = await _use_case(llm).execute(
        [AlignItem("n1", "x", "y", (S1,))], UsageLedger("r", 1.0)
    )
    assert alignment.outcome_id is None and alignment.basis is CpBasis.JUDGE


async def test_weak_candidates_skip_the_model() -> None:
    llm = ScriptedLLM()
    weak = CpCandidate(uuid.uuid4(), None, "tidak relevan", 0.1)
    alignments = await _use_case(llm).execute(
        [AlignItem("n1", "x", "y", (weak,)), AlignItem("n2", "x", "y", ())], UsageLedger("r", 1.0)
    )
    assert [a.basis for a in alignments] == [CpBasis.NO_CANDIDATES, CpBasis.NO_CANDIDATES]
    assert llm.requests == []


async def test_invalid_choice_is_repaired() -> None:
    llm = ScriptedLLM(
        [json.dumps({"choice": "s9", "reason": "x"}), json.dumps({"choice": "s2", "reason": "y"})]
    )
    (alignment,) = await _use_case(llm).execute(
        [AlignItem("n1", "x", "y", (S1, S2))], UsageLedger("r", 1.0)
    )
    assert alignment.outcome_id == S2.outcome_id
    assert 'choice must be one of s1, s2 or "none"' in llm.requests[1].blocks[-1].text
