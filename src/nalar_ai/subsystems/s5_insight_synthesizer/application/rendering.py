"""Prompt blocks for S5. Teacher text is fenced: it is data, never instructions (invariant 5)."""

from collections.abc import Sequence

from nalar_ai.shared.enums import ConceptOutcome
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import ClassCounts
from nalar_ai.subsystems.s5_insight_synthesizer.domain.summary import ParentConcept

OUTCOME_TEXT = {
    ConceptOutcome.MASTERED: "understood",
    ConceptOutcome.DEVELOPING: "still developing",
    ConceptOutcome.MISCONCEPTION: "still holds a mistaken idea",
    ConceptOutcome.NOT_OBSERVED: "not shown in this session",
}


def counts_block(counts: ClassCounts) -> str:
    concepts, misconceptions = counts.concept_aliases(), counts.misconception_aliases()
    lines: list[str] = []
    for c in counts.concepts:
        lines.append(
            f"{concepts.alias(c.id)}: {fence_untrusted('concept', c.name)}\n"
            f"  understood {c.mastered}, developing {c.developing}, "
            f"not observed {c.not_observed}"
        )
        lines.extend(
            f"  {misconceptions.alias(m.id)}: {fence_untrusted('idea', m.statement)}\n"
            f"    holds it now {m.count}, changed their mind during the session "
            f"{m.resolved_count}"
            for m in c.misconceptions
        )
    return "\n".join(lines)


def concepts_block(concepts: Sequence[ParentConcept]) -> str:
    lines: list[str] = []
    for c in concepts:
        line = f"- {fence_untrusted('concept', c.name)}\n  {OUTCOME_TEXT[c.outcome]}"
        if c.misconception:
            line += f"; the idea: {fence_untrusted('idea', c.misconception)}"
        if c.resolved:
            line += "; changed their mind during the session"
        lines.append(line)
    return "\n".join(lines)
