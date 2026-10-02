from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.shared.text import normalize_key


@dataclass(frozen=True, slots=True)
class SourceParagraph:
    id: UUID
    kind: ChunkKind
    content: str
    heading_path: str
    page_start: int | None
    page_end: int | None


def select_paragraphs(
    links: Mapping[UUID, Sequence[UUID]],
    paragraphs: Sequence[SourceParagraph],
    count_tokens: Callable[[str], int],
    token_budget: int,
    per_target: int,
) -> tuple[tuple[SourceParagraph, ...], tuple[UUID, ...]]:
    """Round-robin linked safe paragraphs; deduplicate text and restore reading order."""
    safe = [p for p in paragraphs if p.kind not in (ChunkKind.EXERCISE, ChunkKind.ANSWER_KEY)]
    ordered = sorted(
        safe, key=lambda p: (p.kind != ChunkKind.EXAMPLE, p.kind != ChunkKind.EXPLANATION)
    )
    candidates = {t: [p for p in ordered if p.id in ids] for t, ids in links.items()}
    selected: set[UUID] = set()
    texts: set[str] = set()
    accepted: dict[UUID, int] = dict.fromkeys(links, 0)
    remaining = token_budget
    for _ in range(per_target):
        for target, items in candidates.items():
            while items:
                paragraph = items.pop(0)
                key = normalize_key(paragraph.content)
                if paragraph.id in selected or key in texts:
                    accepted[target] += 1
                    break
                size = count_tokens(paragraph.content + paragraph.heading_path)
                if size > remaining:
                    continue
                selected.add(paragraph.id)
                texts.add(key)
                accepted[target] += 1
                remaining -= size
                break
    return (
        tuple(p for p in paragraphs if p.id in selected),
        tuple(t for t, n in accepted.items() if n == 0),
    )
