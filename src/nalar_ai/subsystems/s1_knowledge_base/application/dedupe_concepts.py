from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import InvalidInputError
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s1_knowledge_base.application.batch import gather_settled
from nalar_ai.subsystems.s1_knowledge_base.domain.dedup_policy import (
    Candidate,
    DedupBasis,
    DedupThresholds,
    decide,
)


class JudgeOut(BaseModel):
    same_as: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class DedupItem:
    key: str
    name: str
    description: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class DedupDecision:
    key: str
    action: Literal["link", "create"]
    existing_concept_id: UUID | None
    basis: DedupBasis
    similarity: float | None
    reason: str | None = None


class DedupeConceptsUseCase:
    PROMPT_ID = "s1.dedup_judge"

    def __init__(self, *, llm: LLMGateway, thresholds: DedupThresholds) -> None:
        self._llm = llm
        self._thresholds = thresholds

    async def execute(self, items: Sequence[DedupItem], ledger: UsageLedger) -> list[DedupDecision]:
        keys = [item.key for item in items]
        if len(set(keys)) != len(keys):
            raise InvalidInputError("item keys must be unique")
        return await gather_settled(self._decide(item, ledger) for item in items)

    async def _decide(self, item: DedupItem, ledger: UsageLedger) -> DedupDecision:
        policy = decide(item.name, item.candidates, self._thresholds)
        if policy.action != "judge":
            assert policy.basis is not None
            return DedupDecision(
                item.key, policy.action, policy.concept_id, policy.basis, policy.similarity
            )
        aliases = AliasMap("e", [c.concept_id for c in policy.judge_candidates])
        listing = "\n".join(
            f"[{aliases.alias(c.concept_id)}] {c.name}: {c.description}"
            for c in policy.judge_candidates
        )
        allowed = " or ".join([", ".join(aliases.aliases()), "null"])
        output = await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "new_concept": fence_untrusted("new_concept", f"{item.name}: {item.description}"),
                "candidates": fence_untrusted("existing_concepts", listing),
            },
            output_model=JudgeOut,
            ledger=ledger,
            retrieval=tuple(
                RetrievalRef(
                    RetrievalSource.CONCEPT, c.concept_id, RetrievalPath.SEARCH, rank, c.similarity
                )
                for rank, c in enumerate(policy.judge_candidates, start=1)
            ),
            semantic_check=lambda out: (
                []
                if out.same_as is None or out.same_as in aliases
                else [f"same_as must be one of {allowed}"]
            ),
        )
        if output.same_as is None:
            return DedupDecision(
                item.key, "create", None, DedupBasis.JUDGE, policy.similarity, output.reason
            )
        chosen = aliases.resolve(output.same_as)
        similarity = next(c.similarity for c in policy.judge_candidates if c.concept_id == chosen)
        return DedupDecision(item.key, "link", chosen, DedupBasis.JUDGE, similarity, output.reason)
