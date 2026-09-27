"""Build domain PageContent directly, for pure domain tests (no PDF needed)."""

from collections.abc import Sequence
from dataclasses import dataclass

from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent, TextLine


@dataclass(frozen=True)
class L:
    text: str
    size: float = 11.0
    bold: bool = False
    gap: float = 0.0  # extra space above this line
    x: float = 72.0


def line(
    text: str,
    *,
    y: float,
    page: int = 1,
    x: float = 72.0,
    size: float = 11.0,
    bold: bool = False,
    width: float | None = None,
) -> TextLine:
    x1 = x + (width if width is not None else len(text) * size * 0.5)
    return TextLine(
        text=text, page=page, x0=x, y0=y, x1=x1, y1=y + size * 1.37, font_size=size, bold=bold
    )


def make_page(
    number: int,
    specs: Sequence[L | str],
    *,
    start_y: float = 100.0,
    extra: Sequence[TextLine] = (),
    image_coverage: float = 0.0,
) -> PageContent:
    lines: list[TextLine] = []
    y = start_y
    for item in specs:
        spec = L(item) if isinstance(item, str) else item
        y += spec.gap
        lines.append(line(spec.text, y=y, page=number, x=spec.x, size=spec.size, bold=spec.bold))
        y += spec.size * 1.5
    return PageContent(
        number=number,
        width=595.0,
        height=842.0,
        lines=tuple(lines) + tuple(extra),
        image_coverage=image_coverage,
    )
