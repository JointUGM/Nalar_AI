"""Heading-bounded, kind-pure, paragraph-packed chunks with no overlap (design doc §7)."""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import groupby

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s1_knowledge_base.domain.sentences import split_sentences
from nalar_ai.subsystems.s1_knowledge_base.domain.structure import Paragraph

HEADING_SEPARATOR = " > "
TokenCount = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    max_tokens: int = 500
    min_tail_tokens: int = 80
    hard_max_tokens: int = 600


DEFAULT_CHUNKING = ChunkingPolicy()


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    local_index: (
        int  # reading order within the section; backend: chunk_index = ordinal*10_000 + this
    )
    content: str
    page_start: int
    page_end: int
    heading_path: str
    kind: ChunkKind
    token_count: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _Piece:
    text: str
    page_start: int
    page_end: int
    tokens: int
    continues: bool  # continues the previous piece's paragraph (join with a space)


def embedding_text(heading_path: str, content: str) -> str:
    return f"{heading_path}\n\n{content}"


def chunk_paragraphs(
    paragraphs: Sequence[Paragraph],
    count_tokens: TokenCount,
    policy: ChunkingPolicy = DEFAULT_CHUNKING,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    for (path, kind), group in groupby(paragraphs, key=lambda p: (p.heading_path, p.kind)):
        heading = HEADING_SEPARATOR.join(path)
        pieces = [piece for paragraph in group for piece in _split(paragraph, count_tokens, policy)]
        for packed in _pack(pieces, count_tokens, policy):
            drafts.append(
                ChunkDraft(
                    local_index=len(drafts),
                    content=packed.text,
                    page_start=packed.page_start,
                    page_end=packed.page_end,
                    heading_path=heading,
                    kind=kind,
                    token_count=packed.tokens,
                    content_sha256=_digest(heading, kind, packed.text),
                )
            )
    return drafts


def _split(paragraph: Paragraph, count: TokenCount, policy: ChunkingPolicy) -> list[_Piece]:
    tokens = count(paragraph.text)
    if tokens <= policy.max_tokens:
        return [_Piece(paragraph.text, paragraph.page_start, paragraph.page_end, tokens, False)]
    units: list[str] = []
    for sentence in split_sentences(paragraph.text):
        if count(sentence) <= policy.max_tokens:
            units.append(sentence)
        else:
            units.extend(_word_windows(sentence, count, policy.max_tokens))
    return [
        _Piece(unit, paragraph.page_start, paragraph.page_end, count(unit), index > 0)
        for index, unit in enumerate(units)
    ]


def _word_windows(text: str, count: TokenCount, limit: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    for word in text.split():
        if current and count(" ".join([*current, word])) > limit:
            windows.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        windows.append(" ".join(current))
    return windows


def _join(pieces: Sequence[_Piece]) -> str:
    text = pieces[0].text
    for piece in pieces[1:]:
        text += (" " if piece.continues else "\n\n") + piece.text
    return text


def _merge(pieces: Sequence[_Piece], count: TokenCount) -> _Piece:
    text = _join(pieces)
    return _Piece(
        text=text,
        page_start=min(piece.page_start for piece in pieces),
        page_end=max(piece.page_end for piece in pieces),
        tokens=count(text),
        continues=pieces[0].continues,
    )


def _pack(pieces: Sequence[_Piece], count: TokenCount, policy: ChunkingPolicy) -> list[_Piece]:
    packed: list[_Piece] = []
    current: list[_Piece] = []
    current_tokens = 0
    for piece in pieces:
        if current and current_tokens + piece.tokens > policy.max_tokens:
            packed.append(_merge(current, count))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += piece.tokens
    if current:
        packed.append(_merge(current, count))
    if len(packed) >= 2 and packed[-1].tokens < policy.min_tail_tokens:
        merged = _merge([packed[-2], packed[-1]], count)
        if merged.tokens <= policy.hard_max_tokens:
            packed[-2:] = [merged]
    return packed


def _digest(heading: str, kind: ChunkKind, text: str) -> str:
    return hashlib.sha256(f"{heading}\n{kind.value}\n{text}".encode()).hexdigest()
