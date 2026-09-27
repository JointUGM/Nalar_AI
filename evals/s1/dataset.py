import json
from dataclasses import dataclass
from pathlib import Path

from nalar_ai.shared.enums import ChunkKind


@dataclass(frozen=True, slots=True)
class GoldConcept:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class GoldMisconception:
    concept: str
    statement: str


@dataclass(frozen=True, slots=True)
class KindLabel:
    page: int
    text_prefix: str
    kind: ChunkKind


@dataclass(frozen=True, slots=True)
class GoldSet:
    topic: str
    concepts: tuple[GoldConcept, ...]
    misconceptions: tuple[GoldMisconception, ...]
    kind_labels: tuple[KindLabel, ...]

    def concept_texts(self) -> list[str]:
        return [f"{c.name}: {c.description}" for c in self.concepts]

    def misconception_statements(self) -> list[str]:
        return [m.statement for m in self.misconceptions]


def load_gold(path: Path) -> GoldSet:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldSet(
        topic=str(raw["topic"]),
        concepts=tuple(GoldConcept(c["name"], c.get("description", "")) for c in raw["concepts"]),
        misconceptions=tuple(
            GoldMisconception(m["concept"], m["statement"]) for m in raw.get("misconceptions", [])
        ),
        kind_labels=tuple(
            KindLabel(int(k["page"]), k["text_prefix"], ChunkKind(k["kind"]))
            for k in raw.get("kind_labels", [])
        ),
    )
