from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nalar_ai.platform.http.context_pack import ContextPackIn, Cue
from nalar_ai.shared.enums import ChunkKind, ProbeStrategy

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
Term = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Levels = Annotated[list[Text], Field(min_length=5, max_length=5)]


class ApprovedConceptIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    name: str = Field(min_length=1, max_length=300)
    description: str = Field(default="", max_length=2000)
    review_status: Literal["approved"] = "approved"
    source_chunk_ids: list[UUID] = Field(default_factory=list, max_length=500)


class ApprovedMisconceptionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    concept_id: UUID
    statement: str = Field(min_length=1, max_length=1000)
    correct_understanding: str = Field(min_length=1, max_length=2000)
    detection_cues: list[Cue] = Field(default_factory=list, max_length=20)
    counter_examples: list[Text] = Field(default_factory=list, max_length=20)
    review_status: Literal["approved"] = "approved"
    source_chunk_ids: list[UUID] = Field(default_factory=list, max_length=500)


class SourceParagraphIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID
    kind: ChunkKind
    content: str = Field(min_length=1, max_length=40000)
    heading_path: str = Field(default="", max_length=2000)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class SelectTargetsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_objective: str = Field(min_length=1, max_length=1000)
    concepts: list[ApprovedConceptIn] = Field(min_length=2, max_length=200)


class SelectTargetsOut(BaseModel):
    target_concept_ids: list[UUID] = Field(min_length=2, max_length=3)


class GenerateMissionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_objective: str = Field(min_length=1, max_length=1000)
    targets: list[ApprovedConceptIn] = Field(min_length=2, max_length=3)
    misconceptions: list[ApprovedMisconceptionIn] = Field(default_factory=list, max_length=30)
    paragraphs: list[SourceParagraphIn] = Field(default_factory=list, max_length=1000)
    max_probes: int = Field(default=6, ge=4, le=6)
    max_duration_minutes: int = Field(default=20, ge=5, le=20)


class MissionRubric(BaseModel):
    claim: Levels
    evidence: Levels
    mechanism: Levels
    transfer: Levels


class GenerateMissionOut(BaseModel):
    context_pack: ContextPackIn
    rubric: MissionRubric
    source_chunk_ids: list[UUID]
    ungrounded_concept_ids: list[UUID]
    probe_plan: dict[str, list[str]]


class SelectionDraft(BaseModel):
    targets: list[str] = Field(min_length=2, max_length=3)


class CoreDraft(BaseModel):
    anchor_problem: str = Field(min_length=1, max_length=4000)
    reference_reasoning: str = Field(min_length=1, max_length=8000)
    rubric: MissionRubric
    answer_terms: list[Term] = Field(min_length=1, max_length=40)


class QuestionDraft(BaseModel):
    move: ProbeStrategy
    text: Text


class BankDraft(BaseModel):
    questions: list[QuestionDraft] = Field(min_length=16, max_length=32)


class QuestionReplacement(BaseModel):
    index: int = Field(ge=0, le=31)
    text: Text


class QuestionRepairs(BaseModel):
    replacements: list[QuestionReplacement] = Field(min_length=1, max_length=32)


class CriticIssue(BaseModel):
    component: Literal["anchor_problem", "reference_reasoning", "rubric", "bank"]
    target: Literal["c1", "c2", "c3"] | None = None
    # Missing means blocking: only an explicit "minor" may survive the second pass.
    severity: Literal["blocking", "minor"] = "blocking"
    problem: Text
    quote: str = Field(min_length=1, max_length=500)


class CriticDraft(BaseModel):
    issues: list[CriticIssue] = Field(max_length=20)
