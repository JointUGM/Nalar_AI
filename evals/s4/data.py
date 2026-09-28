"""S4 eval inputs: finished sessions in the same JSON the backend sends (design doc §13).

    {"context_pack": {...}, "rubric": {...} (optional), "sessions": [
        {"name": "...", "turns": [EvalTurnIn, ...],
         "teacher_levels": {"claim": 2, "evidence": 1, "mechanism": 2, "transfer": 0}}  (optional)
    ]}

Transcripts come from the pilot or from the S3 fake students
(`python -m evals.s3.run personas --transcripts <file>`); the pedagogy lead adds
`teacher_levels` to the ones they score by hand.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from nalar_ai.shared.enums import RubricDimension
from nalar_ai.subsystems.s4_session_evaluator.api.evaluate import (
    EvalTurnIn,
    EvaluationPackIn,
    RubricIn,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.session import EvalSession

DATASETS = Path(__file__).resolve().parent / "datasets"
SAMPLE_SESSIONS = DATASETS / "sample_sessions.json"
SAMPLE_RUBRIC = DATASETS / "sample_rubric.json"


@dataclass(frozen=True, slots=True)
class EvalCase:
    name: str
    session: EvalSession
    teacher_levels: dict[RubricDimension, int] | None


def load_cases(path: Path, rubric_path: Path = SAMPLE_RUBRIC) -> list[EvalCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pack = EvaluationPackIn.model_validate(raw["context_pack"]).to_pack()
    rubric_raw = raw.get("rubric") or json.loads(rubric_path.read_text(encoding="utf-8"))
    rubric = RubricIn.model_validate(rubric_raw).to_rubric()
    cases: list[EvalCase] = []
    for item in raw["sessions"]:
        turns = tuple(EvalTurnIn.model_validate(t).to_turn() for t in item["turns"])
        levels = item.get("teacher_levels")
        cases.append(
            EvalCase(
                name=item["name"],
                session=EvalSession(pack=pack, rubric=rubric, turns=turns),
                teacher_levels=(
                    {RubricDimension(k): int(v) for k, v in levels.items()} if levels else None
                ),
            )
        )
    return cases
