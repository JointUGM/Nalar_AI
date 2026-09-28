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
    TURN_ANALYZE = "turn_analyze"
    PROBE_PLAN = "probe_plan"


class EvalPurpose(StrEnum):
    """Offline eval calls (fake students, judges). Not a DB label: never persisted and never
    returned by an endpoint (InvocationOut refuses them)."""

    FAKE_STUDENT = "eval_fake_student"
    LEAK_JUDGE = "eval_leak_judge"


Purpose = AiPurpose | EvalPurpose


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


class ProbeStrategy(StrEnum):
    """probe_strategy: the five NALAR-Schema.sql labels plus three from migration 002."""

    TRANSFER = "transfer"
    REQUEST_JUSTIFICATION = "request_justification"
    COUNTER_EXAMPLE = "counter_example"
    DECOMPOSE = "decompose"
    REFUSE_AND_REDIRECT = "refuse_and_redirect"
    DEEPER_REASON = "deeper_reason"
    EXPLAIN_MECHANISM = "explain_mechanism"
    SIMPLER_REASON = "simpler_reason"


class PlannerMode(StrEnum):
    TABLE = "table"
    HYBRID = "hybrid"


class MoveSource(StrEnum):
    PLANNER = "planner"
    DEFAULT = "default"
    FIXED_RULE = "fixed_rule"
    PREFILTER = "prefilter"
    FALLBACK_INVALID = "fallback_invalid"
    FALLBACK_ERROR = "fallback_error"


class MoveReasonCode(StrEnum):
    DEFAULT = "default"
    MIXED_ANSWER = "mixed_answer"
    SECOND_WRONG_IDEA = "second_wrong_idea"
    NO_CHANGE_AFTER_EXAMPLE = "no_change_after_example"
    FRUSTRATION = "frustration"
    ALREADY_COVERED = "already_covered"
    CHECK_DEEPER = "check_deeper"


class GuardResult(StrEnum):
    PASSED = "passed"
    BLOCKED_SHAPE = "blocked_shape"
    BLOCKED_VERDICT = "blocked_verdict"
    BLOCKED_NEW_TERMS = "blocked_new_terms"
    BLOCKED_SIMILARITY = "blocked_similarity"
    BLOCKED_DRIFT = "blocked_drift"
    NOT_RUN = "not_run"
