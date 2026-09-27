from nalar_ai.shared.enums import AiPurpose, ChunkKind, RetrievalPath, RetrievalSource


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


def test_s1_ai_purposes_exist_in_migrations_002_and_003() -> None:
    assert {purpose.value for purpose in AiPurpose} == {
        "kb_extract",
        "kb_misconceptions",
        "cp_align",
        "kb_dedup",
        "embedding",
    }


def test_retrieval_labels_match_ai_invocations_retrieval_contract() -> None:
    assert {s.value for s in RetrievalSource} == {
        "material_chunk",
        "cp_outcome",
        "library",
        "concept",
    }
    assert {p.value for p in RetrievalPath} == {"link", "search", "teacher_confirmed"}
