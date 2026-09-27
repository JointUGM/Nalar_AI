"""A failing item must not orphan sibling calls: every paid call stays in the ledger."""

import asyncio
import json
import uuid

import pytest

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.llm.ports import LLMRequest, LLMResponse, LLMUsage, PermanentLLMError
from nalar_ai.platform.prompts.registry import PromptRegistry
from nalar_ai.shared.errors import ModelCallRejectedError
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.subsystems.s1_knowledge_base.application.align_cp import (
    AlignCpUseCase,
    AlignItem,
    CpCandidate,
)
from nalar_ai.subsystems.s1_knowledge_base.application.dedupe_concepts import (
    DedupeConceptsUseCase,
    DedupItem,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.dedup_policy import Candidate, DedupThresholds
from nalar_ai.subsystems.s1_knowledge_base.module import PROMPTS_DIR
from tests.support.s1 import MODELS


class _FirstFailsSecondIsSlow:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if "pertama" in " ".join(block.text for block in request.blocks):
            raise PermanentLLMError("400 bad request")
        await asyncio.sleep(0.01)
        return LLMResponse(text=self.reply, usage=LLMUsage(input_tokens=10, output_tokens=5))


def _gateway(reply: str) -> LLMGateway:
    return LLMGateway(
        port=_FirstFailsSecondIsSlow(reply),
        prompts=PromptRegistry.from_directories([PROMPTS_DIR]),
        models=MODELS,
    )


async def test_dedupe_waits_for_siblings_before_failing() -> None:
    grey = Candidate(uuid.uuid4(), "Gesekan", "d", 0.86)
    ledger = UsageLedger("r", 1.0)
    use_case = DedupeConceptsUseCase(
        llm=_gateway(json.dumps({"same_as": None, "reason": "x"})), thresholds=DedupThresholds()
    )
    with pytest.raises(ModelCallRejectedError):
        await use_case.execute(
            [
                DedupItem("n1", "Konsep pertama", "d", (grey,)),
                DedupItem("n2", "Konsep kedua", "d", (grey,)),
            ],
            ledger,
        )
    assert len(ledger.records) == 2


async def test_align_cp_waits_for_siblings_before_failing() -> None:
    candidate = CpCandidate(uuid.uuid4(), None, "Gaya dan gerak.", 0.7)
    ledger = UsageLedger("r", 1.0)
    use_case = AlignCpUseCase(
        llm=_gateway(json.dumps({"choice": "none", "reason": "x"})), min_similarity=0.25
    )
    with pytest.raises(ModelCallRejectedError):
        await use_case.execute(
            [
                AlignItem("n1", "Konsep pertama", "d", (candidate,)),
                AlignItem("n2", "Konsep kedua", "d", (candidate,)),
            ],
            ledger,
        )
    assert len(ledger.records) == 2
