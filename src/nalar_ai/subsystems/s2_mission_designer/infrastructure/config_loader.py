from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DesignerConfig(BaseModel):
    verdict_terms: list[str] = Field(min_length=1)
    question_max_chars: int = Field(ge=1, le=500)
    question_max_sentences: int = Field(ge=1)
    anchor_max_chars: int = Field(ge=1, le=4000)
    anchor_max_sentences: int = Field(ge=1)


def load_config(path: Path) -> DesignerConfig:
    return DesignerConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
