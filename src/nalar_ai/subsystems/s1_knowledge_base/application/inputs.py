"""Data S1 generation steps receive from the backend (already retrieved from the database)."""

from dataclasses import dataclass
from uuid import UUID

from nalar_ai.shared.enums import ChunkKind


@dataclass(frozen=True, slots=True)
class ChunkRef:
    id: UUID
    heading_path: str
    kind: ChunkKind
    content: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True, slots=True)
class DroppedItem:
    name: str
    reason: str
