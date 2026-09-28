"""The frozen context pack (design doc §5): the S2 → S3 contract, checked before any model call."""

from dataclasses import dataclass
from uuid import UUID

from nalar_ai.shared.enums import ProbeStrategy


@dataclass(frozen=True, slots=True)
class TargetConcept:
    id: UUID
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class MisconceptionRef:
    id: UUID
    concept_id: UUID
    statement: str
    detection_cues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BankQuestion:
    """One teacher-approved question. `id` is stored as session_turns.question_bank_id."""

    id: str
    concept_id: UUID
    move: ProbeStrategy
    text: str
    misconception_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class WriterView:
    """All the question writer may see (AI-1): ids and the approved questions, nothing else.

    No reference reasoning, no concept names or descriptions, no answer terms, and no
    wrong-idea statements either: "Gesekan hanya terjadi pada permukaan kasar" names the
    answer. Concepts and wrong ideas reach the prompt only as aliases.
    """

    target_ids: tuple[UUID, ...]
    misconception_ids: tuple[UUID, ...]
    question_bank: tuple[BankQuestion, ...]


@dataclass(frozen=True, slots=True)
class ContextPack:
    anchor_problem: str
    reference_reasoning: str
    max_probes: int
    max_duration_minutes: int
    targets: tuple[TargetConcept, ...]
    misconceptions: tuple[MisconceptionRef, ...]
    question_bank: tuple[BankQuestion, ...]
    answer_terms: tuple[str, ...]

    @property
    def target_ids(self) -> tuple[UUID, ...]:
        return tuple(target.id for target in self.targets)

    def misconception(self, misconception_id: UUID) -> MisconceptionRef | None:
        return next((m for m in self.misconceptions if m.id == misconception_id), None)

    def writer_view(self) -> WriterView:
        return WriterView(
            target_ids=self.target_ids,
            misconception_ids=tuple(m.id for m in self.misconceptions),
            question_bank=self.question_bank,
        )


def validate_pack(pack: ContextPack) -> list[str]:
    """Problems that make the pack unusable. Empty means valid."""
    problems: list[str] = []
    target_ids = pack.target_ids
    if not target_ids:
        problems.append("the pack has no target concepts")
    if len(set(target_ids)) != len(target_ids):
        problems.append("target concept ids are not unique")
    targets = set(target_ids)

    misconception_ids = [m.id for m in pack.misconceptions]
    if len(set(misconception_ids)) != len(misconception_ids):
        problems.append("misconception ids are not unique")
    concept_of = {m.id: m.concept_id for m in pack.misconceptions}
    for m in pack.misconceptions:
        if m.concept_id not in targets:
            problems.append(f"misconception {m.id} points to a concept that is not a target")

    question_ids = [q.id for q in pack.question_bank]
    if len(set(question_ids)) != len(question_ids):
        problems.append("question bank ids are not unique")
    for q in pack.question_bank:
        if q.concept_id not in targets:
            problems.append(f"question {q.id} points to a concept that is not a target")
        if q.misconception_id is not None and concept_of.get(q.misconception_id) != q.concept_id:
            problems.append(f"question {q.id} names a misconception of another concept")

    covered = {(q.concept_id, q.move) for q in pack.question_bank}
    for target in pack.targets:
        missing = [move.value for move in ProbeStrategy if (target.id, move) not in covered]
        if missing:
            problems.append(
                f"target {target.id} has no approved question for: {', '.join(missing)}"
            )

    if pack.max_probes < len(target_ids):
        problems.append("max_probes is smaller than the number of targets")
    if not pack.answer_terms:
        problems.append("the pack has no answer_terms for the leak guard")
    return problems
