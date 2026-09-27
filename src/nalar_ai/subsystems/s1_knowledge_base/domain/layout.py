"""Layout normalisation for textbook PDFs (design doc §6)."""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from nalar_ai.shared.text import normalize_key
from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent, TextLine

MIN_TEXT_CHARS = 20
SCANNED_MIN_IMAGE_COVERAGE = 0.5

_PAGE_NUMBER = re.compile(r"^(hlm|halaman|page)?\s*\d{1,4}$")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class LayoutPolicy:
    band_ratio: float = 0.08  # top/bottom share of the page where running furniture lives
    repeat_ratio: float = 0.4  # a band line repeated on this share of pages is furniture
    min_pages_for_repeat: int = 3


DEFAULT_LAYOUT = LayoutPolicy()


def _in_band(line: TextLine, page: PageContent, policy: LayoutPolicy) -> bool:
    return line.y0 < page.height * policy.band_ratio or line.y1 > page.height * (
        1 - policy.band_ratio
    )


def _furniture_key(text: str) -> str:
    return _DIGITS.sub("#", normalize_key(text))


def strip_running_furniture(
    pages: Sequence[PageContent], policy: LayoutPolicy = DEFAULT_LAYOUT
) -> list[PageContent]:
    """Remove running headers/footers (repeated band lines) and standalone page numbers."""
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_furniture_key(ln.text) for ln in page.lines if _in_band(ln, page, policy)})
    repeated: set[str] = set()
    if len(pages) >= policy.min_pages_for_repeat:
        threshold = max(2, math.ceil(len(pages) * policy.repeat_ratio))
        repeated = {key for key, count in counts.items() if count >= threshold}

    def is_furniture(line: TextLine, page: PageContent) -> bool:
        if not _in_band(line, page, policy):
            return False
        return bool(_PAGE_NUMBER.match(normalize_key(line.text))) or (
            _furniture_key(line.text) in repeated
        )

    return [
        replace(page, lines=tuple(ln for ln in page.lines if not is_furniture(ln, page)))
        for page in pages
    ]


def reading_order(page: PageContent) -> list[TextLine]:
    """Left column before right column on two-column pages; otherwise top to bottom."""
    lines = list(page.lines)
    if not lines:
        return []
    middle = page.width / 2

    def key(line: TextLine) -> tuple[float, float]:
        return (round(line.y0), line.x0)

    right = [ln for ln in lines if ln.x0 >= middle]
    straddling = [ln for ln in lines if ln.x0 < middle - 10 and ln.x1 > middle + 10]
    two_columns = len(right) >= 0.25 * len(lines) and len(straddling) < 0.10 * len(lines)
    if not two_columns:
        return sorted(lines, key=key)
    left = [ln for ln in lines if ln.x0 < middle]
    return sorted(left, key=key) + sorted(right, key=key)


def join_lines(texts: Sequence[str]) -> str:
    """Join visual lines into running text, repairing words hyphenated across lines."""
    out = ""
    for raw in texts:
        text = raw.strip()
        if not text:
            continue
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and text[:1].islower():
            out = out[:-1] + text
        elif out:
            out = f"{out} {text}"
        else:
            out = text
    return out


def empty_page_reason(page: PageContent) -> Literal["scanned", "blank"] | None:
    """Why a page has no usable text: a scan (image, no text layer) or blank. None if it has text."""
    characters = sum(len(line.text.strip()) for line in page.lines)
    if characters >= MIN_TEXT_CHARS:
        return None
    return "scanned" if page.image_coverage >= SCANNED_MIN_IMAGE_COVERAGE else "blank"
