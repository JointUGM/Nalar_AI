"""The next-lesson teaching plan (TC-23) and its guard. Pure checks, no model.

The plan follows the misconception cycle the research supports: elicit the idea, contrast it
with a reviewed counter-example, explain by refutation, then check in a new situation.
"""

from dataclasses import dataclass

from nalar_ai.shared.text import contains_phrase, normalize_key
from nalar_ai.subsystems.s5_insight_synthesizer.domain.numbers import NumberLexicon

PHASES = ("elicit", "contrast", "explain")


@dataclass(frozen=True, slots=True)
class PlanLimits:
    objective_max_words: int
    action_max_words: int
    question_max_words: int
    look_for_max_words: int

    def __post_init__(self) -> None:
        for field in ("objective", "action", "question", "look_for"):
            value = getattr(self, f"{field}_max_words")
            if type(value) is not int or value < 1:
                raise ValueError(f"{field}_max_words must be an integer >= 1")


@dataclass(frozen=True, slots=True)
class PlanStep:
    phase: str
    teacher_action: str
    ask: str


@dataclass(frozen=True, slots=True)
class TeachingPlan:
    objective: str
    steps: tuple[PlanStep, ...]
    counter_example: int | None  # 1-based index into the supplied reviewed examples
    check_question: str
    look_for: str
    question_ids: tuple[str, ...]


def check_plan(
    plan: TeachingPlan,
    *,
    counter_examples: int,
    question_ids: frozenset[str],
    lexicon: NumberLexicon,
    limits: PlanLimits,
) -> list[str]:
    """Problems phrased for the retry call. Empty means the plan is usable."""
    problems: list[str] = []

    def text(label: str, value: str, max_words: int) -> None:
        if not value.strip():
            problems.append(f"{label}: must not be empty")
        elif len(value.split()) > max_words:
            problems.append(f"{label}: use at most {max_words} words")
        # AI-4: the model gets no counts, so any claim about how many students is invented.
        key = normalize_key(value)
        problems.extend(
            f"{label}: do not say how many students (found '{claim}')"
            for claim in lexicon.count_claims
            if contains_phrase(key, claim)
        )

    text("objective", plan.objective, limits.objective_max_words)
    phases = tuple(step.phase for step in plan.steps)
    if phases != PHASES:
        problems.append(f"steps must be exactly these phases in order: {', '.join(PHASES)}")
    for step in plan.steps:
        text(f"{step.phase} teacher_action", step.teacher_action, limits.action_max_words)
        text(f"{step.phase} ask", step.ask, limits.question_max_words)
    text("check_question", plan.check_question, limits.question_max_words)
    text("look_for", plan.look_for, limits.look_for_max_words)
    if counter_examples and plan.counter_example is None:
        problems.append("counter_example: choose the number of the reviewed example you used")
    elif plan.counter_example is not None and not 1 <= plan.counter_example <= counter_examples:
        problems.append("counter_example: use only a listed reviewed example number, or null")
    unknown = [qid for qid in plan.question_ids if qid not in question_ids]
    if unknown:
        problems.append("question_ids: use only listed question ids")
    return problems
