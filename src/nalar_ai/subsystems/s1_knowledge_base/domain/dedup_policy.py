"""Duplicate policy (design doc §9, §10.3): link, create, or ask the judge."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from uuid import UUID

from nalar_ai.shared.text import normalize_key


class DedupBasis(StrEnum):
    NAME_MATCH = "name_match"
    SIMILARITY = "similarity"
    JUDGE = "judge"
    NO_CANDIDATE = "no_candidate"


@dataclass(frozen=True, slots=True)
class DedupThresholds:
    link_at: float = 0.92
    judge_from: float = 0.82


@dataclass(frozen=True, slots=True)
class Candidate:
    """An existing concept returned by match_kb_concepts, with its cosine similarity."""

    concept_id: UUID
    name: str
    description: str
    similarity: float


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: Literal["link", "create", "judge"]
    basis: DedupBasis | None
    concept_id: UUID | None
    similarity: float | None
    judge_candidates: tuple[Candidate, ...] = ()


def decide(
    name: str,
    candidates: Sequence[Candidate],
    thresholds: DedupThresholds,
    max_judge_candidates: int = 3,
) -> PolicyDecision:
    key = normalize_key(name)
    for candidate in candidates:
        if normalize_key(candidate.name) == key:
            return PolicyDecision(
                "link", DedupBasis.NAME_MATCH, candidate.concept_id, candidate.similarity
            )
    if not candidates:
        return PolicyDecision("create", DedupBasis.NO_CANDIDATE, None, None)
    ranked = sorted(candidates, key=lambda c: (-c.similarity, str(c.concept_id)))
    best = ranked[0]
    if best.similarity >= thresholds.link_at:
        return PolicyDecision("link", DedupBasis.SIMILARITY, best.concept_id, best.similarity)
    grey = [c for c in ranked if c.similarity >= thresholds.judge_from][:max_judge_candidates]
    if grey:
        return PolicyDecision("judge", None, None, best.similarity, tuple(grey))
    return PolicyDecision("create", DedupBasis.SIMILARITY, None, best.similarity)
