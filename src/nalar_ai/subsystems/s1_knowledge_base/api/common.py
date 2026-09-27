from uuid import UUID

from pydantic import BaseModel, Field

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s1_knowledge_base.application.inputs import ChunkRef


class ChunkRefIn(BaseModel):
    """A stored material_chunks row, as the backend read it (ids are real UUIDs)."""

    id: UUID
    heading_path: str = Field(default="", max_length=2000)
    kind: ChunkKind
    content: str = Field(min_length=1, max_length=40_000)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    def to_ref(self) -> ChunkRef:
        return ChunkRef(
            id=self.id,
            heading_path=self.heading_path,
            kind=self.kind,
            content=self.content,
            page_start=self.page_start,
            page_end=self.page_end,
        )


class DroppedOut(BaseModel):
    name: str
    reason: str
