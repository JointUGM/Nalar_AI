"""ContextPackIn: mission_versions.context_pack, the S2 -> S3 contract (design doc §5)."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from nalar_ai.shared.context_pack import (
    BankQuestion,
    ContextPack,
    MisconceptionRef,
    TargetConcept,
)
from nalar_ai.shared.enums import ProbeStrategy

Cue = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Term = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class TargetConceptIn(BaseModel):
    id: UUID
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)


class MisconceptionIn(BaseModel):
    id: UUID
    concept_id: UUID
    statement: str = Field(min_length=1, max_length=1000)
    detection_cues: list[Cue] = Field(default_factory=list, max_length=20)


class BankQuestionIn(BaseModel):
    id: str = Field(min_length=1, max_length=100, description="Stored as question_bank_id")
    concept_id: UUID
    move: ProbeStrategy
    misconception_id: UUID | None = None
    text: str = Field(min_length=1, max_length=500)


class ContextPackIn(BaseModel):
    """The frozen mission package, written before publication and sent on every turn."""

    pack_version: Literal[1]
    anchor_problem: str = Field(min_length=1, max_length=4000)
    reference_reasoning: str = Field(
        min_length=1, max_length=8000, description="Hidden model answer; never sent to the writer"
    )
    max_probes: int = Field(ge=1, le=10, description="mission_versions.max_turns")
    max_duration_minutes: int = Field(ge=1, le=60)
    targets: list[TargetConceptIn] = Field(min_length=1, max_length=5)
    misconceptions: list[MisconceptionIn] = Field(default_factory=list, max_length=30)
    question_bank: list[BankQuestionIn] = Field(min_length=1, max_length=200)
    answer_terms: list[Term] = Field(min_length=1, max_length=50)

    def to_pack(self) -> ContextPack:
        return ContextPack(
            anchor_problem=self.anchor_problem,
            reference_reasoning=self.reference_reasoning,
            max_probes=self.max_probes,
            max_duration_minutes=self.max_duration_minutes,
            targets=tuple(TargetConcept(t.id, t.name, t.description) for t in self.targets),
            misconceptions=tuple(
                MisconceptionRef(m.id, m.concept_id, m.statement, tuple(m.detection_cues))
                for m in self.misconceptions
            ),
            question_bank=tuple(
                BankQuestion(q.id, q.concept_id, q.move, q.text, q.misconception_id)
                for q in self.question_bank
            ),
            answer_terms=tuple(self.answer_terms),
        )
