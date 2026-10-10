from collections.abc import Sequence

COMPONENTS = ("anchor_problem", "reference_reasoning", "rubric", "bank")


def revision_scope(goal_changed: bool, components: Sequence[str]) -> tuple[str, ...]:
    if goal_changed or set(components) & {"anchor_problem", "reference_reasoning"}:
        return COMPONENTS
    return tuple(c for c in COMPONENTS if c in components)
