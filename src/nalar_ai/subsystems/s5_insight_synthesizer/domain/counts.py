"""The class-map counts the backend computed (design doc §4.1). S5 never computes a count."""

from dataclasses import dataclass
from uuid import UUID

from nalar_ai.shared.aliases import AliasMap


@dataclass(frozen=True, slots=True)
class MisconceptionCount:
    id: UUID
    statement: str
    count: int  # holds it now
    resolved_count: int  # changed their mind during the session


@dataclass(frozen=True, slots=True)
class ConceptCount:
    id: UUID
    name: str
    mastered: int
    developing: int
    not_observed: int
    misconceptions: tuple[MisconceptionCount, ...]


@dataclass(frozen=True, slots=True)
class ClassCounts:
    mission_title: str
    total: int  # the denominator: eligible latest attempts
    incomplete: int
    concepts: tuple[ConceptCount, ...]

    def concept_aliases(self) -> AliasMap[UUID]:
        return AliasMap("c", [c.id for c in self.concepts])

    def misconception_aliases(self) -> AliasMap[UUID]:
        return AliasMap("m", [m.id for c in self.concepts for m in c.misconceptions])

    def held_counts(self) -> dict[UUID, int]:
        return {m.id: m.count for c in self.concepts for m in c.misconceptions}

    def names(self) -> tuple[str, ...]:
        """Teacher-written text the number check exempts."""
        return (
            self.mission_title,
            *(c.name for c in self.concepts),
            *(m.statement for c in self.concepts for m in c.misconceptions),
        )


def check_counts(counts: ClassCounts) -> list[str]:
    """Garbage-in checks the DTO bounds can't express. Empty means the counts are usable."""
    problems: list[str] = []
    concept_ids = [c.id for c in counts.concepts]
    if len(set(concept_ids)) != len(concept_ids):
        problems.append("concept ids repeat")
    misconception_ids = [m.id for c in counts.concepts for m in c.misconceptions]
    if len(set(misconception_ids)) != len(misconception_ids):
        problems.append("misconception ids repeat")
    for concept in counts.concepts:
        held = sum(m.count for m in concept.misconceptions)
        if concept.mastered + concept.developing + concept.not_observed + held > counts.total:
            problems.append(f"concept {concept.id}: counts add up to more than the denominator")
        for m in concept.misconceptions:
            if max(m.count, m.resolved_count) > counts.total:
                problems.append(f"misconception {m.id}: a count is above the denominator")
    return list(dict.fromkeys(problems))
