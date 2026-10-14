"""Prompt blocks for S5. Teacher text is fenced: it is data, never instructions (invariant 5)."""

from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import ClassCounts


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
