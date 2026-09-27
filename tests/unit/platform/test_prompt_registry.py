from pathlib import Path

import pytest

from nalar_ai.platform.prompts.registry import PromptError, PromptRegistry, parse_prompt
from nalar_ai.shared.enums import AiPurpose, ModelTier

PROMPT = """---
id: t.echo
version: 1
purpose: kb_extract
tier: quality
max_tokens: 1000
effort: medium
---
=== system ===
You echo words.
=== context ===
Material: {{material}}
=== task ===
Say {{ word }}.
"""


def test_parse_reads_front_matter_and_sections() -> None:
    template = parse_prompt(PROMPT)
    assert template.id == "t.echo"
    assert template.version_tag == "t.echo@v1"
    assert template.purpose is AiPurpose.KB_EXTRACT
    assert template.tier is ModelTier.QUALITY
    assert template.max_tokens == 1000
    assert template.effort == "medium"
    assert template.system == "You echo words."
    assert len(template.sha256) == 64


def test_render_substitutes_once_and_never_rescans_values() -> None:
    rendered = parse_prompt(PROMPT).render({"material": "{{word}} x", "word": "halo"})
    assert rendered.context == "Material: {{word}} x"
    assert rendered.task == "Say halo."


def test_render_rejects_missing_and_unknown_variables() -> None:
    template = parse_prompt(PROMPT)
    with pytest.raises(PromptError, match="missing"):
        template.render({"material": "x"})
    with pytest.raises(PromptError, match="unexpected"):
        template.render({"material": "x", "word": "y", "extra": "z"})


def test_system_section_must_be_static() -> None:
    with pytest.raises(PromptError, match="static"):
        parse_prompt(PROMPT.replace("You echo words.", "You echo {{word}}."))


def test_fast_tier_rejects_effort() -> None:
    with pytest.raises(PromptError, match="effort"):
        parse_prompt(PROMPT.replace("tier: quality", "tier: fast"))


def test_crlf_files_parse() -> None:
    assert parse_prompt(PROMPT.replace("\n", "\r\n")).task == "Say {{ word }}."


def test_registry_loads_directories_and_prefers_highest_version(tmp_path: Path) -> None:
    (tmp_path / "echo.v1.prompt").write_text(PROMPT, encoding="utf-8")
    (tmp_path / "echo.v2.prompt").write_text(
        PROMPT.replace("version: 1", "version: 2"), encoding="utf-8"
    )
    registry = PromptRegistry.from_directories([tmp_path])
    assert registry.ids() == ["t.echo"]
    assert registry.get("t.echo").version == 2
    assert registry.get("t.echo", version=1).version == 1
    with pytest.raises(PromptError, match="unknown prompt"):
        registry.get("t.missing")


def test_registry_rejects_duplicate_versions(tmp_path: Path) -> None:
    (tmp_path / "a.prompt").write_text(PROMPT, encoding="utf-8")
    (tmp_path / "b.prompt").write_text(PROMPT, encoding="utf-8")
    with pytest.raises(PromptError, match="duplicate"):
        PromptRegistry.from_directories([tmp_path])
