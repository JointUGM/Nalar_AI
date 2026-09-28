"""Prompt text for S3. Everything teacher- or student-written is fenced (invariant 5)."""

from collections.abc import Sequence
from uuid import UUID

from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import ProbeStrategy
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import (
    BankQuestion,
    MisconceptionRef,
    TargetConcept,
)
from nalar_ai.subsystems.s3_socratic_prober.domain.session import HistoryTurn

NONE = "(none)"


def clip(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + " …"


def transcript(turns: Sequence[HistoryTurn], *, max_answer_chars: int) -> str:
    if not turns:
        return NONE
    lines: list[str] = []
    for turn in turns:
        lines.append(f"[Q{turn.turn_index}] {turn.question_text.strip()}")
        lines.append(
            f"[A{turn.turn_index}] {clip(turn.answer_text, max_answer_chars) or '(empty)'}"
        )
    return fence_untrusted("session", "\n".join(lines))


def concepts_block(targets: Sequence[TargetConcept], aliases: AliasMap[UUID]) -> str:
    listing = "\n".join(
        f"[{aliases.alias(t.id)}] {t.name}" + (f": {t.description}" if t.description else "")
        for t in targets
    )
    return fence_untrusted("concepts", listing)


def misconceptions_block(
    misconceptions: Sequence[MisconceptionRef],
    concept_aliases: AliasMap[UUID],
    aliases: AliasMap[UUID],
) -> str:
    if not misconceptions:
        return NONE
    lines = []
    for m in misconceptions:
        line = f"[{aliases.alias(m.id)}] ({concept_aliases.alias(m.concept_id)}) {m.statement}"
        if m.detection_cues:
            line += " | students say: " + "; ".join(m.detection_cues)
        lines.append(line)
    return fence_untrusted("wrong_ideas", "\n".join(lines))


def bank_block(
    target_ids: Sequence[UUID],
    bank: Sequence[BankQuestion],
    concept_aliases: AliasMap[UUID],
    question_aliases: AliasMap[str],
    misconception_aliases: AliasMap[UUID],
) -> str:
    lines: list[str] = []
    for target in target_ids:
        lines.append(f"{concept_aliases.alias(target)}:")
        for move in ProbeStrategy:
            questions = [q for q in bank if q.concept_id == target and q.move is move]
            if not questions:
                continue
            lines.append(f"  {move.value}:")
            for q in questions:
                tag = (
                    f" (for {misconception_aliases.alias(q.misconception_id)})"
                    if q.misconception_id is not None
                    else ""
                )
                lines.append(f"    [{question_aliases.alias(q.id)}]{tag} {q.text}")
    return fence_untrusted("question_bank", "\n".join(lines))
