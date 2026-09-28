"""S4 gate metrics (design doc §13)."""

from collections.abc import Mapping, Sequence

from nalar_ai.shared.enums import RubricDimension

GATES = {"consistency": 0.90, "teacher_within_one": 0.90}

Levels = Mapping[RubricDimension, int]


def consistency(runs_per_session: Sequence[Sequence[Levels]]) -> float | None:
    """Share of (session, dimension) pairs whose level was identical in every run.
    None when no session was scored at least twice."""
    agree = total = 0
    for runs in runs_per_session:
        if len(runs) < 2:
            continue
        for dimension in RubricDimension:
            total += 1
            agree += len({run[dimension] for run in runs}) == 1
    return agree / total if total else None


def teacher_agreement(pairs: Sequence[tuple[Levels, Levels]]) -> dict[str, float] | None:
    """(model, teacher) level pairs: exact and within-one agreement over every dimension."""
    diffs = [
        abs(model[dimension] - teacher[dimension])
        for model, teacher in pairs
        for dimension in RubricDimension
    ]
    if not diffs:
        return None
    return {
        "exact": sum(d == 0 for d in diffs) / len(diffs),
        "within_one": sum(d <= 1 for d in diffs) / len(diffs),
    }
