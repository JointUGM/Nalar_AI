"""Regenerate openapi.json, the contract nalar-backend generates its typed client from.

Run: uv run python scripts/export_openapi.py
"""

import json
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from nalar_ai.container import build_container
from nalar_ai.main import create_app
from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.settings import Settings

OPENAPI_PATH = Path(__file__).resolve().parents[1] / "openapi.json"


def build_openapi() -> dict[str, Any]:
    settings = Settings.model_construct(
        service_key=SecretStr("openapi-export"), llm_provider="fake"
    )
    container = build_container(settings, llm_port=ScriptedLLM(), embedding_port=HashingEmbedder())
    return create_app(container).openapi()


def render() -> str:
    return json.dumps(build_openapi(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    OPENAPI_PATH.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {OPENAPI_PATH}")


if __name__ == "__main__":
    main()
