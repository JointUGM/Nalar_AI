"""The class-map insight and its guard (design doc §5, §6). Pure checks, no model."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import ClassCounts
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import (
    NumberLexicon,
    number_problems,
)
from nalar_ai.subsystems.s5_insight_synthesizer.domain.placeholders import (
    has_braces,
    rewrite,
    strip_placeholders,
)


@dataclass(frozen=True, slots=True)
class InsightLimits:
    narrative_max_words: int
    explanation_max_words: int
    name_max_words: int
    max_clusters: int


@dataclass(frozen=True, slots=True)
class DraftCluster:
    name: str
    explanation: str
    misconceptions: tuple[str, ...]  # aliases, as the model wrote them


@dataclass(frozen=True, slots=True)
class Cluster:
    name: str
    explanation: str
    misconception_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class Insight:
    narrative: str  # placeholders keyed by real ids (design doc §4.2)
    clusters: tuple[Cluster, ...]


def check_insight(
    narrative: str,
    clusters: Sequence[DraftCluster],
    counts: ClassCounts,
    lexicon: NumberLexicon,
    limits: InsightLimits,
) -> tuple[Insight | None, list[str]]:
    """The insight with ids, or None and the problems phrased for the retry call."""
    concepts, misconceptions = counts.concept_aliases(), counts.misconception_aliases()
    held, names = counts.held_counts(), counts.names()
    problems: list[str] = []

    def words(label: str, value: str, max_words: int) -> None:
        if not value.strip():
            problems.append(f"{label}: must not be empty")
        elif len(value.split()) > max_words:
            problems.append(f"{label}: use at most {max_words} words")

    def with_placeholders(label: str, value: str, max_words: int) -> str:
        rewritten, found = rewrite(value, concepts, misconceptions)
        found += number_problems(strip_placeholders(value), lexicon, names)
        problems.extend(f"{label}: {problem}" for problem in found)
        words(label, value, max_words)
        return rewritten.strip()

    out_narrative = with_placeholders("narrative", narrative, limits.narrative_max_words)
    if len(clusters) > limits.max_clusters:
        problems.append(f"use at most {limits.max_clusters} clusters")
    seen: set[str] = set()
    out_clusters: list[Cluster] = []
    for number, draft in enumerate(clusters, start=1):
        label = f"cluster {number}"
        if has_braces(draft.name):
            problems.append(f"{label} name: no placeholders in a name")
        else:
            problems.extend(
                f"{label} name: {p}" for p in number_problems(draft.name, lexicon, names)
            )
        words(f"{label} name", draft.name, limits.name_max_words)
        explanation = with_placeholders(
            f"{label} explanation", draft.explanation, limits.explanation_max_words
        )
        if not draft.misconceptions:
            problems.append(f"{label}: name at least one idea")
        ids: list[UUID] = []
        for written in draft.misconceptions:
            alias = written.strip().casefold()
            if alias not in misconceptions:
                problems.append(f"{label}: unknown idea '{written[:20]}'")
            elif held[misconceptions.resolve(alias)] == 0:
                problems.append(f"{label}: no student holds {alias} now; leave it out")
            elif alias in seen:
                problems.append(f"{label}: {alias} is already in another cluster")
            else:
                seen.add(alias)
                ids.append(misconceptions.resolve(alias))
        out_clusters.append(Cluster(draft.name.strip(), explanation, tuple(ids)))
    if problems:
        return None, list(dict.fromkeys(problems))
    return Insight(out_narrative, tuple(out_clusters)), []
