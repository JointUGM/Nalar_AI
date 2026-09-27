"""Chapters from the PDF outline (design doc §5)."""

from collections.abc import Callable
from dataclasses import dataclass

from nalar_ai.subsystems.s1_knowledge_base.domain.models import PdfOutline


@dataclass(frozen=True, slots=True)
class DetectedSection:
    ordinal: int
    level: int
    title: str
    page_start: int
    page_end: int  # inclusive; chapters often start mid-page, so the chunk step trims by title
    parent_ordinal: int | None
    suggested: bool  # pre-tick in the UI: level-1, not front or back matter


def detect_sections(
    outline: PdfOutline,
    *,
    fallback_title: str,
    is_excluded_title: Callable[[str], bool],
    max_level: int = 2,
) -> list[DetectedSection]:
    entries = [e for e in outline.toc if 1 <= e.page <= outline.page_count and e.title.strip()]
    if not entries:
        return [DetectedSection(1, 1, fallback_title.strip(), 1, outline.page_count, None, True)]
    top = min(entry.level for entry in entries)
    entries = [entry for entry in entries if entry.level - top + 1 <= max_level]
    sections: list[DetectedSection] = []
    latest_at_level: dict[int, int] = {}
    for index, entry in enumerate(entries):
        level = entry.level - top + 1
        page_end = outline.page_count
        for later in entries[index + 1 :]:
            if later.level <= entry.level:
                page_end = max(entry.page, later.page)
                break
        ordinal = index + 1
        parent = latest_at_level.get(level - 1) if level > 1 else None
        latest_at_level = {lvl: o for lvl, o in latest_at_level.items() if lvl < level}
        latest_at_level[level] = ordinal
        sections.append(
            DetectedSection(
                ordinal=ordinal,
                level=level,
                title=entry.title.strip(),
                page_start=entry.page,
                page_end=page_end,
                parent_ordinal=parent,
                suggested=level == 1 and not is_excluded_title(entry.title),
            )
        )
    return sections
