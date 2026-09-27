"""Misconceptions per concept (design doc §10.5), one Sonnet call per concept, in parallel."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import InvalidInputError, OutputValidationError
from nalar_ai.shared.provenance import RetrievalRef, UsageLedger
from nalar_ai.shared.text import fence_untrusted, normalize_key
from nalar_ai.shared.vectors import cosine
from nalar_ai.subsystems.s1_knowledge_base.application.inputs import ChunkRef, DroppedItem
from nalar_ai.subsystems.s1_knowledge_base.application.material import (
    chunk_retrieval,
    render_material,
    select_evidence,
)

MAX_MISCONCEPTIONS = 4
MAX_LIBRARY_CANDIDATES = 5


class MisconceptionOut(BaseModel):
    statement: str
    correct_understanding: str
    student_phrasings: list[str]
    counter_examples: list[str]
    source_chunks: list[str]
    library: str | None


class MisconceptionsOut(BaseModel):
    misconceptions: list[MisconceptionOut]


@dataclass(frozen=True, slots=True)
class LibraryCandidate:
    """A published misconception_library row returned by match_misconception_library."""

    library_id: UUID
    statement: str
    correct_understanding: str
    student_phrasings: tuple[str, ...]
    counter_examples: tuple[str, ...]
    similarity: float


@dataclass(frozen=True, slots=True)
class ConceptForMisconceptions:
    concept_ref: str
    name: str
    description: str
    source_chunk_ids: tuple[UUID, ...]
    library_candidates: tuple[LibraryCandidate, ...] = ()
    existing_statements: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerateMisconceptionsCommand:
    subject: str
    phase: str
    chunks: tuple[ChunkRef, ...]
    concepts: tuple[ConceptForMisconceptions, ...]


@dataclass(frozen=True, slots=True)
class MisconceptionDraft:
    concept_ref: str
    statement: str
    correct_understanding: str
    detection_cues: tuple[str, ...]
    counter_examples: tuple[str, ...]
    source_chunk_ids: tuple[UUID, ...]
    library_id: UUID | None
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FailedConcept:
    concept_ref: str
    error: str


@dataclass(frozen=True, slots=True)
class GenerateMisconceptionsResult:
    misconceptions: list[MisconceptionDraft]
    dropped: list[DroppedItem]
    failed: list[FailedConcept]
    embedding_model: str


def validate_misconceptions(
    output: MisconceptionsOut, chunk_aliases: AliasMap[UUID], library_aliases: AliasMap[UUID]
) -> list[str]:
    errors: list[str] = []
    count = len(output.misconceptions)
    if not 1 <= count <= MAX_MISCONCEPTIONS:
        errors.append(f"return between 1 and {MAX_MISCONCEPTIONS} misconceptions, got {count}")
    seen: set[str] = set()
    for index, item in enumerate(output.misconceptions, start=1):
        label = f"misconception {index}"
        statement_key = normalize_key(item.statement)
        if not statement_key:
            errors.append(f"{label}: statement is empty")
        if statement_key == normalize_key(item.correct_understanding):
            errors.append(f"{label}: statement and correct_understanding must differ")
        if statement_key in seen:
            errors.append(f"{label}: repeats an earlier statement")
        seen.add(statement_key)
        if len([p for p in item.student_phrasings if p.strip()]) < 2:
            errors.append(f"{label}: give at least 2 student_phrasings")
        if not any(example.strip() for example in item.counter_examples):
            errors.append(f"{label}: give at least 1 counter_example")
        if not item.source_chunks:
            errors.append(f"{label}: needs at least one source chunk")
        errors.extend(
            f"{label}: unknown chunk {alias}"
            for alias in item.source_chunks
            if alias not in chunk_aliases
        )
        if item.library is not None and item.library not in library_aliases:
            errors.append(f"{label}: unknown library entry {item.library}")
    return errors


def _render_library(entries: Sequence[LibraryCandidate], aliases: AliasMap[UUID]) -> str:
    if not entries:
        return "(none)"
    blocks = [
        f"[{aliases.alias(e.library_id)}] {e.statement}\n"
        f"  correct idea: {e.correct_understanding}\n"
        f"  students say: {'; '.join(e.student_phrasings) or '-'}\n"
        f"  counter-examples: {'; '.join(e.counter_examples) or '-'}"
        for e in entries
    ]
    return fence_untrusted("misconception_library", "\n".join(blocks))


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


class GenerateMisconceptionsUseCase:
    PROMPT_ID = "s1.generate_misconceptions"

    def __init__(
        self,
        *,
        llm: LLMGateway,
        embeddings: EmbeddingService,
        estimate_tokens: Callable[[str], int],
        evidence_budget_tokens: int,
        duplicate_threshold: float,
    ) -> None:
        self._llm = llm
        self._embeddings = embeddings
        self._estimate_tokens = estimate_tokens
        self._budget = evidence_budget_tokens
        self._threshold = duplicate_threshold

    async def execute(
        self, command: GenerateMisconceptionsCommand, ledger: UsageLedger
    ) -> GenerateMisconceptionsResult:
        refs = [concept.concept_ref for concept in command.concepts]
        if len(set(refs)) != len(refs):
            raise InvalidInputError("concept_ref values must be unique")
        outcomes = await asyncio.gather(
            *(self._one(command, concept, ledger) for concept in command.concepts),
            return_exceptions=True,
        )
        misconceptions: list[MisconceptionDraft] = []
        dropped: list[DroppedItem] = []
        failed: list[FailedConcept] = []
        for concept, outcome in zip(command.concepts, outcomes, strict=True):
            if isinstance(outcome, OutputValidationError | InvalidInputError):
                failed.append(FailedConcept(concept.concept_ref, outcome.message))
            elif isinstance(outcome, BaseException):
                raise outcome  # provider down or budget exceeded: the whole request fails
            else:
                misconceptions.extend(outcome[0])
                dropped.extend(outcome[1])
        return GenerateMisconceptionsResult(misconceptions, dropped, failed, self._embeddings.model)

    async def _one(
        self,
        command: GenerateMisconceptionsCommand,
        concept: ConceptForMisconceptions,
        ledger: UsageLedger,
    ) -> tuple[list[MisconceptionDraft], list[DroppedItem]]:
        evidence = select_evidence(
            command.chunks, concept.source_chunk_ids, self._estimate_tokens, self._budget
        )
        if not evidence:
            raise InvalidInputError(f"no usable source chunks for concept {concept.concept_ref}")
        chunk_aliases = AliasMap("c", [chunk.id for chunk in evidence])
        library = sorted(concept.library_candidates, key=lambda e: -e.similarity)
        library = library[:MAX_LIBRARY_CANDIDATES]
        library_aliases = AliasMap("L", [entry.library_id for entry in library])
        existing = "\n".join(f"- {s}" for s in concept.existing_statements)
        output = await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "subject": command.subject,
                "phase": command.phase,
                "material": render_material(evidence, chunk_aliases),
                "concept": fence_untrusted("concept", f"{concept.name}: {concept.description}"),
                "library": _render_library(library, library_aliases),
                "existing": fence_untrusted("existing_misconceptions", existing)
                if existing
                else "(none)",
            },
            output_model=MisconceptionsOut,
            ledger=ledger,
            retrieval=chunk_retrieval(evidence)
            + tuple(
                RetrievalRef(
                    RetrievalSource.LIBRARY, e.library_id, RetrievalPath.SEARCH, rank, e.similarity
                )
                for rank, e in enumerate(library, start=1)
            ),
            semantic_check=lambda out: validate_misconceptions(out, chunk_aliases, library_aliases),
        )
        texts = [item.statement for item in output.misconceptions]
        texts += list(concept.existing_statements)
        vectors = await self._embeddings.embed(texts, tag="misconception", ledger=ledger)
        count = len(output.misconceptions)
        new_vectors, existing_vectors = vectors[:count], vectors[count:]
        drafts: list[MisconceptionDraft] = []
        dropped: list[DroppedItem] = []
        kept: list[tuple[float, ...]] = []
        for item, vector in zip(output.misconceptions, new_vectors, strict=True):
            if any(cosine(vector, other) >= self._threshold for other in existing_vectors):
                dropped.append(DroppedItem(item.statement, "duplicate_of_existing"))
                continue
            if any(cosine(vector, other) >= self._threshold for other in kept):
                dropped.append(DroppedItem(item.statement, "duplicate_in_batch"))
                continue
            kept.append(vector)
            drafts.append(
                MisconceptionDraft(
                    concept_ref=concept.concept_ref,
                    statement=item.statement.strip(),
                    correct_understanding=item.correct_understanding.strip(),
                    detection_cues=_unique(item.student_phrasings),
                    counter_examples=_unique(item.counter_examples),
                    source_chunk_ids=tuple(
                        dict.fromkeys(chunk_aliases.resolve(a) for a in item.source_chunks)
                    ),
                    library_id=library_aliases.resolve(item.library) if item.library else None,
                    embedding=vector,
                )
            )
        return drafts, dropped
