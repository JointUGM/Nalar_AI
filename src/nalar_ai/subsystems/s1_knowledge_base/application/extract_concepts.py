"""Concept extraction (design doc §10.2): Sonnet lists a chapter's key concepts; code validates."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from uuid import UUID

from pydantic import BaseModel

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import InvalidInputError, SectionTooLargeError
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger
from nalar_ai.shared.text import fence_untrusted, normalize_key
from nalar_ai.shared.vectors import cosine
from nalar_ai.subsystems.s1_knowledge_base.application.inputs import ChunkRef, DroppedItem
from nalar_ai.subsystems.s1_knowledge_base.application.material import (
    chunk_retrieval,
    eligible_chunks,
    render_material,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.graph import find_cycle

MAX_CONCEPTS = 15
MIN_CONCEPTS_BEFORE_WARNING = 6
MAX_NAME_CHARS = 80
_NEW_KEY = re.compile(r"n\d{1,3}")


class ConceptOut(BaseModel):
    key: str
    name: str
    description: str
    source_chunks: list[str]
    prerequisites: list[str]


class ExistingMentionOut(BaseModel):
    existing: str
    source_chunks: list[str]


class ExtractionOut(BaseModel):
    concepts: list[ConceptOut]
    existing_mentions: list[ExistingMentionOut]


@dataclass(frozen=True, slots=True)
class ExistingConcept:
    id: UUID
    name: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class PrerequisiteEdge:
    concept_id: UUID
    prerequisite_concept_id: UUID


@dataclass(frozen=True, slots=True)
class RejectedConcept:
    name: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class ExtractConceptsCommand:
    subject: str
    phase: str
    section_title: str
    chunks: tuple[ChunkRef, ...]
    existing_concepts: tuple[ExistingConcept, ...] = ()
    existing_prerequisites: tuple[PrerequisiteEdge, ...] = ()
    rejected_concepts: tuple[RejectedConcept, ...] = ()


@dataclass(frozen=True, slots=True)
class ConceptDraft:
    key: str
    name: str
    description: str
    source_chunk_ids: tuple[UUID, ...]
    prerequisite_keys: tuple[str, ...]
    prerequisite_existing_ids: tuple[UUID, ...]
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExistingLink:
    concept_id: UUID
    source_chunk_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class ExtractConceptsResult:
    concepts: list[ConceptDraft]
    existing_links: list[ExistingLink]
    dropped: list[DroppedItem]
    warnings: list[str]
    embedding_model: str


def concept_text(name: str, description: str) -> str:
    """What a concept's embedding represents (used by dedup, CP and library lookups)."""
    return f"{name.strip()}: {description.strip()}"


def validate_extraction(
    output: ExtractionOut,
    chunk_aliases: AliasMap[UUID],
    existing_aliases: AliasMap[UUID],
    existing_names: Mapping[str, str],
    existing_edges: Sequence[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    if not output.concepts and not output.existing_mentions:
        errors.append("return at least one concept or existing mention")
    if len(output.concepts) > MAX_CONCEPTS:
        errors.append(f"return at most {MAX_CONCEPTS} new concepts, got {len(output.concepts)}")
    keys = [concept.key.strip() for concept in output.concepts]
    for duplicate in sorted({key for key in keys if keys.count(key) > 1}):
        errors.append(f"duplicate key {duplicate}")
    seen_names: dict[str, str] = {}
    for concept, key in zip(output.concepts, keys, strict=True):
        if not _NEW_KEY.fullmatch(key):
            errors.append(f"{key}: key must look like n1, n2, ...")
        name_key = normalize_key(concept.name)
        if not name_key:
            errors.append(f"{key}: name is empty")
        elif len(concept.name) > MAX_NAME_CHARS:
            errors.append(f"{key}: name is longer than {MAX_NAME_CHARS} characters")
        if name_key in existing_names:
            errors.append(
                f"{key}: '{concept.name}' already exists as {existing_names[name_key]}; "
                "put it in existing_mentions"
            )
        elif name_key and name_key in seen_names:
            errors.append(f"{key}: same name as {seen_names[name_key]}")
        seen_names.setdefault(name_key, key)
        if not concept.description.strip():
            errors.append(f"{key}: description is empty")
        if not concept.source_chunks:
            errors.append(f"{key}: needs at least one source chunk")
        errors.extend(
            f"{key}: unknown chunk {a}" for a in concept.source_chunks if a not in chunk_aliases
        )
        for ref in (r.strip() for r in concept.prerequisites):
            if ref == key:
                errors.append(f"{key}: cannot be its own prerequisite")
            elif ref not in keys and ref not in existing_aliases:
                errors.append(f"{key}: unknown prerequisite {ref}")
    for mention in output.existing_mentions:
        if mention.existing not in existing_aliases:
            errors.append(f"existing_mentions: unknown concept {mention.existing}")
        if not mention.source_chunks:
            errors.append(f"existing_mentions {mention.existing}: needs at least one source chunk")
        errors.extend(
            f"existing_mentions {mention.existing}: unknown chunk {a}"
            for a in mention.source_chunks
            if a not in chunk_aliases
        )
    new_edges = [
        (key, ref.strip())
        for concept, key in zip(output.concepts, keys, strict=True)
        for ref in concept.prerequisites
        if ref.strip() != key
    ]
    cycle = find_cycle([*existing_edges, *new_edges])
    if cycle is not None:
        errors.append("prerequisites form a cycle: " + " -> ".join(cycle))
    return errors


@dataclass
class _Accepted:
    concept: ConceptOut
    vector: tuple[float, ...]
    sources: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)


class ExtractConceptsUseCase:
    PROMPT_ID = "s1.extract_concepts"

    def __init__(
        self,
        *,
        llm: LLMGateway,
        embeddings: EmbeddingService,
        estimate_tokens: Callable[[str], int],
        max_section_tokens: int,
        duplicate_threshold: float,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._estimate_tokens = estimate_tokens
        self._max_section_tokens = max_section_tokens
        self._threshold = duplicate_threshold

    async def execute(
        self, command: ExtractConceptsCommand, ledger: UsageLedger
    ) -> ExtractConceptsResult:
        chunks = eligible_chunks(command.chunks)
        if not chunks:
            raise InvalidInputError("the section has no chunks usable for concept extraction")
        chunk_aliases = AliasMap("c", [chunk.id for chunk in chunks])
        existing_aliases = AliasMap("k", [concept.id for concept in command.existing_concepts])
        material = render_material(chunks, chunk_aliases)
        estimated = self._estimate_tokens(material)
        if estimated > self._max_section_tokens:
            raise SectionTooLargeError(
                f"the section is about {estimated} tokens; build its subsections instead",
                details={"estimated_tokens": estimated, "limit": self._max_section_tokens},
            )
        existing_ids = {concept.id for concept in command.existing_concepts}
        existing_names = {
            normalize_key(c.name): existing_aliases.alias(c.id) for c in command.existing_concepts
        }
        existing_edges = [
            (
                existing_aliases.alias(e.concept_id),
                existing_aliases.alias(e.prerequisite_concept_id),
            )
            for e in command.existing_prerequisites
            if e.concept_id in existing_ids and e.prerequisite_concept_id in existing_ids
        ]
        existing_listing = "\n".join(
            f"[{existing_aliases.alias(c.id)}] {c.name}: {c.description}"
            for c in command.existing_concepts
        )
        rejected_listing = "\n".join(f"- {r.name}" for r in command.rejected_concepts)
        variables = {
            "subject": fence_untrusted("subject", command.subject),
            "phase": fence_untrusted("phase", command.phase),
            "section_title": fence_untrusted("section_title", command.section_title),
            "material": material,
            "existing_concepts": fence_untrusted("existing_concepts", existing_listing)
            if existing_listing
            else "(none)",
            "rejected_concepts": fence_untrusted("rejected_concepts", rejected_listing)
            if rejected_listing
            else "(none)",
            "max_concepts": str(MAX_CONCEPTS),
        }
        retrieval = chunk_retrieval(chunks) + tuple(
            RetrievalRef(RetrievalSource.CONCEPT, concept.id, RetrievalPath.LINK, rank)
            for rank, concept in enumerate(command.existing_concepts, start=1)
        )
        output = await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables=variables,
            output_model=ExtractionOut,
            ledger=ledger,
            retrieval=retrieval,
            semantic_check=lambda out: validate_extraction(
                out, chunk_aliases, existing_aliases, existing_names, existing_edges
            ),
        )
        return await self._finish(
            output, command, chunk_aliases, existing_aliases, existing_edges, ledger
        )

    async def _finish(
        self,
        output: ExtractionOut,
        command: ExtractConceptsCommand,
        chunk_aliases: AliasMap[UUID],
        existing_aliases: AliasMap[UUID],
        existing_edges: list[tuple[str, str]],
        ledger: UsageLedger,
    ) -> ExtractConceptsResult:
        warnings: list[str] = []
        if len(output.concepts) < MIN_CONCEPTS_BEFORE_WARNING:
            warnings.append(f"only {len(output.concepts)} new concepts were extracted")
        rejected = command.rejected_concepts
        texts = [concept_text(c.name, c.description) for c in output.concepts]
        texts += [concept_text(r.name, r.description) for r in rejected]
        vectors = await self._embeddings.embed(texts, tag="concept", ledger=ledger) if texts else []
        new_vectors, rejected_vectors = (
            vectors[: len(output.concepts)],
            vectors[len(output.concepts) :],
        )
        rejected_names = {normalize_key(r.name) for r in rejected}

        dropped: list[DroppedItem] = []
        accepted: dict[str, _Accepted] = {}
        redirect: dict[str, str | None] = {}
        for concept, vector in zip(output.concepts, new_vectors, strict=True):
            key = concept.key.strip()
            if normalize_key(concept.name) in rejected_names or any(
                cosine(vector, other) >= self._threshold for other in rejected_vectors
            ):
                dropped.append(DroppedItem(concept.name, "matches_rejected"))
                redirect[key] = None
                continue
            twin = next(
                (a for a in accepted.values() if cosine(vector, a.vector) >= self._threshold), None
            )
            if twin is not None:
                twin_key = twin.concept.key.strip()
                dropped.append(DroppedItem(concept.name, f"merged_into:{twin_key}"))
                redirect[key] = twin_key
                twin.sources.extend(concept.source_chunks)
                twin.prerequisites.extend(concept.prerequisites)
                continue
            accepted[key] = _Accepted(
                concept, vector, list(concept.source_chunks), list(concept.prerequisites)
            )

        new_edges: list[tuple[str, str]] = []
        existing_prerequisites: dict[str, list[UUID]] = {}
        for key, item in accepted.items():
            for ref in dict.fromkeys(r.strip() for r in item.prerequisites):
                if ref in existing_aliases:
                    existing_prerequisites.setdefault(key, []).append(existing_aliases.resolve(ref))
                    continue
                target = redirect.get(ref, ref)
                if target is None or target == key or target not in accepted:
                    continue
                if (key, target) not in new_edges:
                    new_edges.append((key, target))
        new_edges, removed = _break_cycles(existing_edges, new_edges)
        if removed:
            warnings.append(
                f"removed {removed} prerequisite link(s) that formed a cycle after merging"
            )

        drafts = [
            ConceptDraft(
                key=key,
                name=item.concept.name.strip(),
                description=item.concept.description.strip(),
                source_chunk_ids=tuple(
                    dict.fromkeys(chunk_aliases.resolve(a) for a in item.sources)
                ),
                prerequisite_keys=tuple(target for source, target in new_edges if source == key),
                prerequisite_existing_ids=tuple(dict.fromkeys(existing_prerequisites.get(key, []))),
                embedding=item.vector,
            )
            for key, item in accepted.items()
        ]
        mention_sources: dict[str, list[str]] = {}
        for mention in output.existing_mentions:
            mention_sources.setdefault(mention.existing.strip(), []).extend(mention.source_chunks)
        existing_links = [
            ExistingLink(
                concept_id=existing_aliases.resolve(alias),
                source_chunk_ids=tuple(dict.fromkeys(chunk_aliases.resolve(a) for a in sources)),
            )
            for alias, sources in mention_sources.items()
        ]
        return ExtractConceptsResult(
            concepts=drafts,
            existing_links=existing_links,
            dropped=dropped,
            warnings=warnings,
            embedding_model=self._embeddings.model,
        )


def _break_cycles(
    existing: Sequence[tuple[str, str]], new: list[tuple[str, str]]
) -> tuple[list[tuple[str, str]], int]:
    """Merging near-duplicates can close a cycle; drop the newest new edge on each cycle."""
    edges = list(new)
    removed = 0
    while (cycle := find_cycle([*existing, *edges])) is not None:
        victim = next(
            (pair for pair in reversed(list(pairwise(cycle))) if pair in edges),
            None,
        )
        if victim is None:
            break  # a cycle made only of existing edges is not ours to fix
        edges.remove(victim)
        removed += 1
    return edges, removed
