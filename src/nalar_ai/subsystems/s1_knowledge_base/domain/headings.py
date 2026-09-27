"""Heading detection from font size and weight."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from nalar_ai.subsystems.s1_knowledge_base.domain.models import TextLine

_TRAILING_PUNCTUATION = (".", ",", ";", ":")


@dataclass(frozen=True, slots=True)
class HeadingPolicy:
    size_ratio: float = 1.15  # font this much larger than body text is a heading
    max_heading_chars: int = 120
    max_bold_heading_words: int = 12
    max_size_levels: int = 3


DEFAULT_HEADINGS = HeadingPolicy()


def body_font_size(lines: Sequence[TextLine]) -> float:
    """The font size carrying the most characters (ties go to the smaller size)."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.font_size, 1)] += len(line.text)
    if not weights:
        return 11.0
    return max(weights.items(), key=lambda item: (item[1], -item[0]))[0]


def _could_be_heading(text: str, policy: HeadingPolicy) -> bool:
    text = text.strip()
    return (
        bool(text)
        and len(text) <= policy.max_heading_chars
        and any(character.isalpha() for character in text)
    )


class HeadingDetector:
    """Larger fonts become levels 1..3 by size; bold body-size short lines are the deepest level."""

    def __init__(self, lines: Sequence[TextLine], policy: HeadingPolicy = DEFAULT_HEADINGS) -> None:
        self._policy = policy
        self._body = body_font_size(lines)
        large = sorted(
            {
                round(ln.font_size, 1)
                for ln in lines
                if ln.font_size >= self._body * policy.size_ratio
                and _could_be_heading(ln.text, policy)
            },
            reverse=True,
        )
        self._size_levels = {
            size: min(rank + 1, policy.max_size_levels) for rank, size in enumerate(large)
        }
        self._bold_level = min(len(set(self._size_levels.values())) + 1, 4)

    @property
    def body_size(self) -> float:
        return self._body

    def level(self, line: TextLine) -> int | None:
        text = line.text.strip()
        if not _could_be_heading(text, self._policy):
            return None
        size = round(line.font_size, 1)
        if size in self._size_levels:
            return self._size_levels[size]
        if (
            line.bold
            and len(text.split()) <= self._policy.max_bold_heading_words
            and not text.endswith(_TRAILING_PUNCTUATION)
        ):
            return self._bold_level
        return None
