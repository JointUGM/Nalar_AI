"""Lines → paragraphs, each with a heading path and a kind (design doc §7.2–7.4)."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.shared.text import normalize_key
from nalar_ai.subsystems.s1_knowledge_base.domain.headings import (
    DEFAULT_HEADINGS,
    HeadingDetector,
    HeadingPolicy,
)
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import KindLexicon, KindMarker, RegionScope
from nalar_ai.subsystems.s1_knowledge_base.domain.layout import join_lines, reading_order
from nalar_ai.subsystems.s1_knowledge_base.domain.models import PageContent, TextLine

MAX_HEADING_LEVEL = 4
_BULLET = re.compile(r"^([•▪●◦\-–]|\d{1,2}[.)]|[a-z][.)])\s")
_ANSWER_START = re.compile(r"^(jawab|jawaban|penyelesaian|pembahasan)\s*:", re.IGNORECASE)
_CHAPTER_LABEL = re.compile(r"bab [\divxlc]+")
_MIN_PARTIAL_TITLE_CHARS = 8


@dataclass(frozen=True, slots=True)
class Paragraph:
    text: str
    page_start: int
    page_end: int
    heading_path: tuple[str, ...]
    kind: ChunkKind


@dataclass(frozen=True, slots=True)
class StructurePolicy:
    paragraph_gap_ratio: float = 0.8  # vertical gap (x body size) that starts a new paragraph
    block_end_gap_ratio: float = 2.5  # vertical gap (x body size) that ends a block region


DEFAULT_STRUCTURE = StructurePolicy()


def title_matches(line_text: str, title: str) -> bool:
    """Whether a printed line is (part of) a section title, e.g. "Bab 2" or "Tekanan Zat"."""
    line_key, title_key = normalize_key(line_text), normalize_key(title)
    if not line_key or not title_key:
        return False
    if line_key == title_key:
        return True
    if _CHAPTER_LABEL.fullmatch(line_key):
        return title_key.startswith(line_key + " ")
    if len(line_key) < _MIN_PARTIAL_TITLE_CHARS:
        return False
    return (
        title_key.startswith(line_key)
        or title_key.endswith(line_key)
        or line_key.startswith(title_key)
    )


def build_paragraphs(
    pages: Sequence[PageContent],
    *,
    section_title: str,
    next_section_title: str | None,
    lexicon: KindLexicon,
    headings: HeadingPolicy = DEFAULT_HEADINGS,
    policy: StructurePolicy = DEFAULT_STRUCTURE,
) -> list[Paragraph]:
    lines = [line for page in pages for line in reading_order(page)]
    lines = _trim_to_section(lines, section_title, next_section_title)
    if not lines:
        return []
    builder = _Builder(section_title.strip(), lexicon, HeadingDetector(lines, headings), policy)
    for line in lines:
        builder.feed(line)
    return builder.finish()


def _trim_to_section(lines: list[TextLine], title: str, next_title: str | None) -> list[TextLine]:
    """Chapters can start mid-page: drop text before our title and from the next title on."""
    if not lines:
        return lines
    first_page = lines[0].page
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if ln.page == first_page and title_matches(ln.text, title)
        ),
        0,
    )
    lines = lines[start:]
    if next_title and lines:
        last_page = lines[-1].page
        stop = next(
            (
                i
                for i, ln in enumerate(lines)
                if i > 0 and ln.page == last_page and title_matches(ln.text, next_title)
            ),
            None,
        )
        if stop is not None:
            lines = lines[:stop]
    return lines


class _Builder:
    def __init__(
        self,
        root: str,
        lexicon: KindLexicon,
        detector: HeadingDetector,
        policy: StructurePolicy,
    ) -> None:
        self._root = root
        self._lexicon = lexicon
        self._detector = detector
        self._policy = policy
        self._stack: list[tuple[int, str]] = []
        self._region: tuple[int, KindMarker] | None = None
        self._region_has_body = False
        self._buffer: list[TextLine] = []
        self._previous: TextLine | None = None
        self._paragraphs: list[Paragraph] = []

    def feed(self, line: TextLine) -> None:
        text = line.text.strip()
        marker = self._lexicon.match(text, bold=line.bold)
        level = self._detector.level(line)
        if marker is not None:
            self._flush()
            region_level = min(
                level if level is not None else self._deepest() + 1, MAX_HEADING_LEVEL
            )
            self._push(region_level, text)
            self._region = (region_level, marker)
            self._region_has_body = False
        elif level is not None:
            self._flush()
            if not self._echoes_root(text):
                self._close_region_for_heading(level)
                self._push(level, text)
        else:
            self._feed_body(line)
            return
        self._previous = line

    def finish(self) -> list[Paragraph]:
        self._flush()
        return self._paragraphs

    def _feed_body(self, line: TextLine) -> None:
        if self._block_region_ends_by_gap(line):
            self._flush()
            self._close_block_region()
        elif self._starts_new_paragraph(line):
            self._flush()
        self._buffer.append(line)
        if self._region is not None:
            self._region_has_body = True
        self._previous = line

    def _block_region_ends_by_gap(self, line: TextLine) -> bool:
        if self._region is None or self._region[1].scope is not RegionScope.BLOCK:
            return False
        previous = self._previous
        if previous is None or not self._region_has_body or line.page != previous.page:
            return False
        gap = line.y0 - previous.y1
        return gap > self._policy.block_end_gap_ratio * self._detector.body_size

    def _starts_new_paragraph(self, line: TextLine) -> bool:
        if not self._buffer:
            return False
        last = self._buffer[-1]
        if line.page != last.page:
            return False  # a page break never ends a paragraph by itself
        if abs(line.font_size - last.font_size) > 0.5 or _BULLET.match(line.text.strip()):
            return True
        return line.y0 - last.y1 > self._policy.paragraph_gap_ratio * self._detector.body_size

    def _close_region_for_heading(self, level: int) -> None:
        if self._region is None:
            return
        region_level, marker = self._region
        # Restrictive (section) kinds persist through same-level sub-headings.
        ends = level <= region_level if marker.scope is RegionScope.BLOCK else level < region_level
        if ends:
            self._region = None

    def _close_block_region(self) -> None:
        if self._region is None:
            return
        region_level = self._region[0]
        self._stack = [(level, title) for level, title in self._stack if level < region_level]
        self._region = None

    def _push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))

    def _deepest(self) -> int:
        return self._stack[-1][0] if self._stack else 0

    def _echoes_root(self, text: str) -> bool:
        return title_matches(text, self._root) or bool(
            _CHAPTER_LABEL.fullmatch(normalize_key(text))
        )

    def _flush(self) -> None:
        if not self._buffer:
            return
        text = join_lines([line.text for line in self._buffer])
        kind = self._region[1].kind if self._region is not None else ChunkKind.EXPLANATION
        if kind is ChunkKind.EXERCISE and _ANSWER_START.match(text):
            kind = ChunkKind.ANSWER_KEY
        self._paragraphs.append(
            Paragraph(
                text=text,
                page_start=self._buffer[0].page,
                page_end=self._buffer[-1].page,
                heading_path=(self._root, *(title for _, title in self._stack)),
                kind=kind,
            )
        )
        self._buffer.clear()
