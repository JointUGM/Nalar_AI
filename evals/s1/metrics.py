from collections.abc import Sequence
from dataclasses import dataclass

from evals.s1.dataset import KindLabel
from nalar_ai.shared.enums import ChunkKind
from nalar_ai.shared.text import normalize_key
from nalar_ai.shared.vectors import cosine

RESTRICTED = frozenset({ChunkKind.EXERCISE, ChunkKind.ANSWER_KEY})

# A chunk for kind evaluation: (page_start, page_end, content, kind)
ChunkRow = tuple[int, int, str, ChunkKind]


@dataclass(frozen=True, slots=True)
class RecallReport:
    matched: int
    total: int
    missing: tuple[str, ...]

    @property
    def recall(self) -> float:
        return self.matched / self.total if self.total else 0.0


def match_recall(
    gold: Sequence[str],
    predicted: Sequence[str],
    gold_vectors: Sequence[Sequence[float]],
    predicted_vectors: Sequence[Sequence[float]],
    threshold: float,
) -> RecallReport:
    """A gold item is found if a prediction has the same normalised text or cosine >= threshold."""
    predicted_keys = {normalize_key(text) for text in predicted}
    missing: list[str] = []
    for text, vector in zip(gold, gold_vectors, strict=True):
        found = normalize_key(text) in predicted_keys or any(
            cosine(vector, other) >= threshold for other in predicted_vectors
        )
        if not found:
            missing.append(text)
    return RecallReport(len(gold) - len(missing), len(gold), tuple(missing))


@dataclass(frozen=True, slots=True)
class KindReport:
    accuracy: float
    restricted_recall: float  # exercise/answer_key labels found AND predicted as restricted
    unmatched: tuple[KindLabel, ...]  # labels whose text was not found in any chunk


def kind_report(labels: Sequence[KindLabel], chunks: Sequence[ChunkRow]) -> KindReport:
    correct = restricted_total = restricted_hit = 0
    unmatched: list[KindLabel] = []
    for label in labels:
        prefix = normalize_key(label.text_prefix)
        chunk = next(
            (c for c in chunks if c[0] <= label.page <= c[1] and prefix in normalize_key(c[2])),
            None,
        )
        if chunk is None:
            unmatched.append(label)
            # An exercise or answer key the chunker never produced is a miss for the safety
            # gate, not something to leave out of it.
            restricted_total += label.kind in RESTRICTED
            continue
        correct += chunk[3] is label.kind
        if label.kind in RESTRICTED:
            restricted_total += 1
            restricted_hit += chunk[3] in RESTRICTED
    return KindReport(
        accuracy=correct / len(labels) if labels else 0.0,
        restricted_recall=restricted_hit / restricted_total if restricted_total else 0.0,
        unmatched=tuple(unmatched),
    )


def recall_at_k(ranked: Sequence[Sequence[str]], gold: Sequence[set[str]], k: int) -> float:
    """Share of queries whose top-k contains at least one gold id (piece lookup, CP recall)."""
    if not ranked:
        return 0.0
    hits = sum(1 for ids, wanted in zip(ranked, gold, strict=True) if wanted & set(ids[:k]))
    return hits / len(ranked)
