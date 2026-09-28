"""Prompt blocks for S4. Teacher- and student-written text is fenced by the callers and here:
it is data, never instructions (invariant 5)."""

from collections.abc import Sequence
from uuid import UUID

from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s4_session_evaluator.domain.session import (
    EvalMisconception,
    EvalTarget,
    EvalTurn,
    Rubric,
)


def clip(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + " …"


def rubric_block(rubric: Rubric) -> str:
    lines: list[str] = []
    for dimension, descriptors in rubric.descriptors.items():
        lines.append(f"{dimension.value}:")
        lines.extend(f"  {level}: {text}" for level, text in enumerate(descriptors))
    return "\n".join(lines)


def concepts_block(targets: Sequence[EvalTarget], concepts: AliasMap[UUID]) -> str:
    return "\n".join(
        f"{concepts.alias(t.id)}: {t.name}" + (f" ({t.description})" if t.description else "")
        for t in targets
    )


def misconceptions_block(
    misconceptions: Sequence[EvalMisconception],
    concepts: AliasMap[UUID],
    aliases: AliasMap[UUID],
) -> str:
    if not misconceptions:
        return "(none listed)"
    lines = []
    for m in misconceptions:
        cues = "; ".join(m.detection_cues)
        line = f"{aliases.alias(m.id)} (concept {concepts.alias(m.concept_id)}): {m.statement}"
        lines.append(line + (f" Students say: {cues}" if cues else ""))
    return "\n".join(lines)


def transcript(
    turns: Sequence[EvalTurn], concepts: AliasMap[UUID], *, max_answer_chars: int
) -> str:
    """Every turn with its alias; the question's move and target help find the transfer probe."""
    blocks: list[str] = []
    for turn in turns:
        about = (
            f"{turn.move.value}, concept {concepts.alias(turn.target_concept_id)}"
            if turn.move is not None and turn.target_concept_id is not None
            else "opening question"
        )
        answer = (
            fence_untrusted("student_answer", clip(turn.answer_text, max_answer_chars))
            if turn.answered
            else "(no answer)"
        )
        question = fence_untrusted("question", turn.question_text)
        blocks.append(f"{turn.alias} ({about})\nQuestion:\n{question}\nAnswer:\n{answer}")
    return "\n\n".join(blocks)
