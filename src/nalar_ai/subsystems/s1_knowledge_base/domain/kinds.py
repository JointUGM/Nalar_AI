"""Chunk kinds from textbook markers (design doc §7.4). Kind is a leak-safety boundary:
exercise and answer_key text must never be labelled as anything else."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.shared.text import normalize_key

# The only kinds concept extraction and misconception generation may read.
GENERATION_KINDS: frozenset[ChunkKind] = frozenset(
    {
        ChunkKind.EXPLANATION,
        ChunkKind.EXAMPLE,
        ChunkKind.ACTIVITY,
        ChunkKind.SUMMARY,
        ChunkKind.SIDEBAR,
    }
)

MAX_MARKER_WORDS = 8
_NUMBERING = re.compile(r"(bab )?[\divxlc]+( [\divxlc]+)*")


class RegionScope(StrEnum):
    BLOCK = "block"  # ends at a heading at the same or a higher level, or a large vertical gap
    SECTION = "section"  # ends only at a heading at a higher level


@dataclass(frozen=True, slots=True)
class KindMarker:
    kind: ChunkKind
    scope: RegionScope
    marker: str


@dataclass(frozen=True, slots=True)
class KindLexicon:
    version: int
    markers: tuple[KindMarker, ...]  # normalised, longest first

    @classmethod
    def build(cls, version: int, markers: Iterable[KindMarker]) -> "KindLexicon":
        normalised = [KindMarker(m.kind, m.scope, normalize_key(m.marker)) for m in markers]
        if any(m.kind is ChunkKind.EXPLANATION for m in normalised):
            raise ValueError("explanation is the default kind and cannot have markers")
        ordered = sorted(normalised, key=lambda m: (-len(m.marker), m.marker))
        return cls(version=version, markers=tuple(ordered))

    def match(self, text: str, *, bold: bool) -> KindMarker | None:
        key = normalize_key(text)
        if not key or len(key.split()) > MAX_MARKER_WORDS:
            return None
        for marker in self.markers:
            if key == marker.marker:
                return marker
            if key.startswith(marker.marker + " "):
                rest = key[len(marker.marker) + 1 :]
                if bold or _NUMBERING.fullmatch(rest):
                    return marker
        return None

    def is_front_or_back_matter(self, title: str) -> bool:
        marker = self.match(title, bold=True)
        return marker is not None and marker.kind is ChunkKind.OTHER
