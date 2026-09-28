"""Eval inputs (formats in evals/s3/datasets/README.md, local only)."""

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import yaml

from nalar_ai.subsystems.s3_socratic_prober.api.pack import ContextPackIn
from nalar_ai.subsystems.s3_socratic_prober.domain.moves import AnswerType
from nalar_ai.subsystems.s3_socratic_prober.domain.pack import ContextPack

SAMPLE_PACK = Path(__file__).resolve().parent / "datasets" / "sample_pack.json"
PERSONAS = Path(__file__).resolve().parent / "personas.yaml"


@dataclass(frozen=True, slots=True)
class LabelledAnswer:
    question: str
    answer: str
    answer_type: AnswerType
    misconception_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class StudentScript:
    """A scripted student: answers in order, the last one repeated when they run out."""

    name: str
    kind: str  # adversarial | steering | normal
    answers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Persona:
    name: str
    description: str


def load_pack(path: Path = SAMPLE_PACK) -> ContextPack:
    """The same JSON the backend stores in mission_versions.context_pack."""
    return ContextPackIn.model_validate_json(path.read_text(encoding="utf-8")).to_pack()


def load_labelled(path: Path) -> list[LabelledAnswer]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        LabelledAnswer(
            question=item["question"],
            answer=item["answer"],
            answer_type=AnswerType(item["answer_type"]),
            misconception_ids=tuple(UUID(m) for m in item.get("misconception_ids", [])),
        )
        for item in raw["items"]
    ]


def load_scripts(path: Path) -> list[StudentScript]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    scripts = [
        StudentScript(name=s["name"], kind=s["kind"], answers=tuple(s["answers"]))
        for s in raw["scripts"]
    ]
    if any(not s.answers for s in scripts):
        raise ValueError("every script needs at least one answer")
    return scripts


def load_personas(path: Path = PERSONAS) -> list[Persona]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Persona(name=p["name"], description=p["description"].strip()) for p in raw["personas"]]
