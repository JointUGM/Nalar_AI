from nalar_ai.subsystems.s2_mission_designer.domain.revision import revision_scope


def test_goal_change_invalidates_all_pedagogical_components():
    assert revision_scope(True, ("rubric",)) == (
        "anchor_problem",
        "reference_reasoning",
        "rubric",
        "bank",
    )


def test_rubric_revision_does_not_expand_to_bank():
    assert revision_scope(False, ("rubric",)) == ("rubric",)


def test_reference_change_invalidates_existing_questions():
    assert revision_scope(False, ("reference_reasoning",)) == (
        "anchor_problem",
        "reference_reasoning",
        "rubric",
        "bank",
    )
