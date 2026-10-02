from nalar_ai.shared.enums import (
    AiPurpose,
    ChunkKind,
    ConceptOutcome,
    GuardResult,
    MoveReasonCode,
    MoveSource,
    PlannerMode,
    ProbeStrategy,
    RetrievalPath,
    RetrievalSource,
    RubricDimension,
)


def test_chunk_kind_labels_match_migration_003() -> None:
    assert [kind.value for kind in ChunkKind] == [
        "explanation",
        "example",
        "activity",
        "exercise",
        "answer_key",
        "summary",
        "sidebar",
        "other",
    ]


def test_ai_purposes_exist_in_the_schema_and_migrations_002_and_003() -> None:
    assert {purpose.value for purpose in AiPurpose} == {
        "mission_generation",
        "mission_critic",
        "kb_extract",
        "kb_misconceptions",
        "cp_align",
        "kb_dedup",
        "embedding",
        "turn_analyze",
        "probe_plan",
        "session_evaluation",
        "reflection_generation",
        "class_map_insight",
        "parent_summary",
    }


def test_prober_labels_match_migration_002() -> None:
    assert [m.value for m in ProbeStrategy] == [
        "transfer",
        "request_justification",
        "counter_example",
        "decompose",
        "refuse_and_redirect",
        "deeper_reason",
        "explain_mechanism",
        "simpler_reason",
    ]
    assert [m.value for m in PlannerMode] == ["table", "hybrid"]
    assert [m.value for m in MoveSource] == [
        "planner",
        "default",
        "fixed_rule",
        "prefilter",
        "fallback_invalid",
        "fallback_error",
    ]
    assert [m.value for m in MoveReasonCode] == [
        "default",
        "mixed_answer",
        "second_wrong_idea",
        "no_change_after_example",
        "frustration",
        "already_covered",
        "check_deeper",
    ]
    assert [m.value for m in GuardResult] == [
        "passed",
        "blocked_shape",
        "blocked_verdict",
        "blocked_new_terms",
        "blocked_similarity",
        "blocked_drift",
        "not_run",
    ]


def test_retrieval_labels_match_ai_invocations_retrieval_contract() -> None:
    assert {s.value for s in RetrievalSource} == {
        "material_chunk",
        "cp_outcome",
        "library",
        "concept",
    }
    assert {p.value for p in RetrievalPath} == {"link", "search", "teacher_confirmed"}


def test_evaluation_labels_match_the_base_schema() -> None:
    assert [d.value for d in RubricDimension] == ["claim", "evidence", "mechanism", "transfer"]
    assert [o.value for o in ConceptOutcome] == [
        "mastered",
        "developing",
        "misconception",
        "not_observed",
    ]
