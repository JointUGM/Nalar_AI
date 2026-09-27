"""Mirrors of database enum labels. Values must match the migrations exactly."""

from enum import StrEnum


class ChunkKind(StrEnum):
    EXPLANATION = "explanation"
    EXAMPLE = "example"
    ACTIVITY = "activity"
    EXERCISE = "exercise"
    ANSWER_KEY = "answer_key"
    SUMMARY = "summary"
    SIDEBAR = "sidebar"
    OTHER = "other"


class AiPurpose(StrEnum):
    KB_EXTRACT = "kb_extract"
    KB_MISCONCEPTIONS = "kb_misconceptions"
    CP_ALIGN = "cp_align"
    KB_DEDUP = "kb_dedup"
    EMBEDDING = "embedding"


class RetrievalSource(StrEnum):
    MATERIAL_CHUNK = "material_chunk"
    CP_OUTCOME = "cp_outcome"
    LIBRARY = "library"
    CONCEPT = "concept"


class RetrievalPath(StrEnum):
    LINK = "link"
    SEARCH = "search"
    TEACHER_CONFIRMED = "teacher_confirmed"


class CallStatus(StrEnum):
    """Mapped by the backend onto its ai_call_status labels."""

    SUCCESS = "success"
    ERROR = "error"


class ModelTier(StrEnum):
    FAST = "fast"  # Haiku 4.5
    QUALITY = "quality"  # Sonnet 5
    JUDGE = "judge"  # Opus 5, evals only
