"""CP alignment (design doc §10.4): Haiku picks one CP statement among retrieved candidates."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import RetrievalPath, RetrievalSource
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger
from nalar_ai.shared.text import fence_untrusted

MAX_CANDIDATES = 5


class CpBasis(StrEnum):
    JUDGE = "judge"
    NO_CANDIDATES = "no_candidates"


class CpChoiceOut(BaseModel):
    choice: str
    reason: str


@dataclass(frozen=True, slots=True)
class CpCandidate:
    """A statement-grain row returned by match_cp_outcomes."""

    outcome_id: UUID
    element: str | None
    description: str
    similarity: float


@dataclass(frozen=True, slots=True)
class AlignItem:
    concept_ref: str
    name: str
    description: str
    candidates: tuple[CpCandidate, ...]


@dataclass(frozen=True, slots=True)
class CpAlignment:
    concept_ref: str
    outcome_id: UUID | None
    basis: CpBasis
    reason: str


class AlignCpUseCase:
    PROMPT_ID = "s1.cp_align"

    def __init__(self, *, llm: LLMGateway, min_similarity: float) -> None:
        self._llm = llm
        self._min_similarity = min_similarity

    async def execute(self, items: Sequence[AlignItem], ledger: UsageLedger) -> list[CpAlignment]:
        return list(await asyncio.gather(*(self._align(item, ledger) for item in items)))

    async def _align(self, item: AlignItem, ledger: UsageLedger) -> CpAlignment:
        candidates = sorted(item.candidates, key=lambda c: (-c.similarity, str(c.outcome_id)))
        candidates = candidates[:MAX_CANDIDATES]
        if not candidates or candidates[0].similarity < self._min_similarity:
            return CpAlignment(
                item.concept_ref, None, CpBasis.NO_CANDIDATES, "no CP statement is similar enough"
            )
        aliases = AliasMap("s", [c.outcome_id for c in candidates])
        listing = "\n".join(
            f"[{aliases.alias(c.outcome_id)}] ({c.element or '-'}) {c.description}"
            for c in candidates
        )
        allowed = f'{", ".join(aliases.aliases())} or "none"'
        output = await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "concept": fence_untrusted("concept", f"{item.name}: {item.description}"),
                "statements": fence_untrusted("cp_statements", listing),
            },
            output_model=CpChoiceOut,
            ledger=ledger,
            retrieval=tuple(
                RetrievalRef(
                    RetrievalSource.CP_OUTCOME,
                    c.outcome_id,
                    RetrievalPath.SEARCH,
                    rank,
                    c.similarity,
                )
                for rank, c in enumerate(candidates, start=1)
            ),
            semantic_check=lambda out: (
                []
                if out.choice.strip() == "none" or out.choice in aliases
                else [f"choice must be one of {allowed}"]
            ),
        )
        if output.choice.strip() == "none":
            return CpAlignment(item.concept_ref, None, CpBasis.JUDGE, output.reason)
        return CpAlignment(
            item.concept_ref, aliases.resolve(output.choice), CpBasis.JUDGE, output.reason
        )
