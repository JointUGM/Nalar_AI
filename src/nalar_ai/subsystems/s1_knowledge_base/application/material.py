"""How teacher material enters prompts: eligible kinds only, aliased, fenced, traced."""

from collections.abc import Callable, Sequence
from uuid import UUID

from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import RetrievalPath, RetrievalSource
from nalar_ai.shared.provenance import RetrievalRef
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s1_knowledge_base.application.inputs import ChunkRef
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import GENERATION_KINDS


def eligible_chunks(chunks: Sequence[ChunkRef]) -> list[ChunkRef]:
    """Exercises, answer keys and other never reach a generation prompt (AI-13)."""
    return [chunk for chunk in chunks if chunk.kind in GENERATION_KINDS]


def _pages(chunk: ChunkRef) -> str:
    if chunk.page_start is None:
        return "hlm. ?"
    if chunk.page_end is None or chunk.page_end == chunk.page_start:
        return f"hlm. {chunk.page_start}"
    return f"hlm. {chunk.page_start}-{chunk.page_end}"


def render_material(chunks: Sequence[ChunkRef], aliases: AliasMap[UUID]) -> str:
    blocks = [
        f"[{aliases.alias(chunk.id)}] ({_pages(chunk)}, {chunk.kind.value}) {chunk.heading_path}\n"
        f"{chunk.content}"
        for chunk in chunks
    ]
    return fence_untrusted("teacher_material", "\n\n".join(blocks))


def chunk_retrieval(chunks: Sequence[ChunkRef]) -> tuple[RetrievalRef, ...]:
    return tuple(
        RetrievalRef(RetrievalSource.MATERIAL_CHUNK, chunk.id, RetrievalPath.LINK, rank)
        for rank, chunk in enumerate(chunks, start=1)
    )


def select_evidence(
    chunks: Sequence[ChunkRef],
    source_ids: Sequence[UUID],
    estimate_tokens: Callable[[str], int],
    budget: int,
) -> list[ChunkRef]:
    """Evidence pack for one concept (design doc §10.5): its source chunks first, then their
    neighbours in reading order, within a token budget. Returned in reading order."""
    eligible = eligible_chunks(chunks)
    position = {chunk.id: index for index, chunk in enumerate(eligible)}
    primary = [position[source] for source in dict.fromkeys(source_ids) if source in position]
    neighbours = [j for i in primary for j in (i - 1, i + 1) if 0 <= j < len(eligible)]
    chosen: list[int] = []
    used = 0
    for index in dict.fromkeys([*primary, *neighbours]):
        cost = estimate_tokens(eligible[index].content)
        if chosen and used + cost > budget:
            continue
        chosen.append(index)
        used += cost
    return [eligible[index] for index in sorted(chosen)]
