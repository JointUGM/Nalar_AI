"""Versioned prompt files (`*.prompt`, plain text; markdown files never go to git).

Format:
    ---
    id: s1.extract_concepts
    version: 1
    purpose: kb_extract          # ai_purpose label
    tier: quality                # fast | quality | judge
    max_tokens: 16000
    effort: high                 # optional; never on the fast tier (Haiku 4.5 rejects it)
    ---
    === system ===               # static text only: identical on every call, so cacheable
    === context ===              # optional; large per-call reference data, cached when possible
    === task ===                 # the per-call instruction

Variables are written {{name}}. Rendering substitutes once and never rescans values, so
untrusted text can't inject variables.
"""

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nalar_ai.shared.enums import AiPurpose, ModelTier
from nalar_ai.shared.errors import ConfigurationError

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_SECTION = re.compile(r"^=== (system|context|task) ===$", re.MULTILINE)
_VARIABLE = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class PromptError(ConfigurationError):
    code = "prompt_error"


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    template: "PromptTemplate"
    system: str
    context: str | None
    task: str


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    id: str
    version: int
    purpose: AiPurpose
    tier: ModelTier
    max_tokens: int
    effort: str | None
    system: str
    context: str | None
    task: str
    sha256: str

    @property
    def version_tag(self) -> str:
        return f"{self.id}@v{self.version}"

    def variables(self) -> set[str]:
        text = "\n".join(part for part in (self.system, self.context or "", self.task))
        return set(_VARIABLE.findall(text))

    def render(self, variables: Mapping[str, str]) -> RenderedPrompt:
        expected = self.variables()
        missing = expected - variables.keys()
        unexpected = variables.keys() - expected
        if missing:
            raise PromptError(f"{self.version_tag}: missing variables {sorted(missing)}")
        if unexpected:
            raise PromptError(f"{self.version_tag}: unexpected variables {sorted(unexpected)}")

        def substitute(text: str) -> str:
            return _VARIABLE.sub(lambda match: variables[match.group(1)], text)

        return RenderedPrompt(
            template=self,
            system=substitute(self.system),
            context=substitute(self.context) if self.context is not None else None,
            task=substitute(self.task),
        )


def parse_prompt(text: str) -> PromptTemplate:
    text = text.replace("\r\n", "\n")
    front = _FRONT_MATTER.match(text)
    if front is None:
        raise PromptError("prompt file is missing its --- front matter ---")
    meta: dict[str, Any] = yaml.safe_load(front.group(1)) or {}
    parts = _SECTION.split(text[front.end() :])
    sections = dict(zip(parts[1::2], (part.strip() for part in parts[2::2]), strict=True))
    prompt_id = str(meta.get("id", ""))
    if "system" not in sections or "task" not in sections:
        raise PromptError(f"{prompt_id}: needs '=== system ===' and '=== task ===' sections")
    tier = ModelTier(meta["tier"])
    effort = meta.get("effort")
    if effort is not None and effort not in _EFFORTS:
        raise PromptError(f"{prompt_id}: unknown effort {effort!r}")
    if effort is not None and tier is ModelTier.FAST:
        raise PromptError(f"{prompt_id}: effort is not supported on the fast tier (Haiku 4.5)")
    if _VARIABLE.search(sections["system"]):
        raise PromptError(f"{prompt_id}: the system section must be static (no variables)")
    return PromptTemplate(
        id=prompt_id,
        version=int(meta["version"]),
        purpose=AiPurpose(meta["purpose"]),
        tier=tier,
        max_tokens=int(meta["max_tokens"]),
        effort=effort,
        system=sections["system"],
        context=sections.get("context"),
        task=sections["task"],
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


class PromptRegistry:
    def __init__(self, templates: Iterable[PromptTemplate]) -> None:
        self._templates: dict[str, dict[int, PromptTemplate]] = {}
        for template in templates:
            versions = self._templates.setdefault(template.id, {})
            if template.version in versions:
                raise PromptError(f"duplicate prompt {template.version_tag}")
            versions[template.version] = template

    @classmethod
    def from_directories(cls, directories: Iterable[Path]) -> "PromptRegistry":
        templates: list[PromptTemplate] = []
        for directory in directories:
            if not directory.is_dir():
                raise PromptError(f"prompt directory not found: {directory}")
            for path in sorted(directory.glob("*.prompt")):
                templates.append(parse_prompt(path.read_text(encoding="utf-8")))
        return cls(templates)

    def get(self, prompt_id: str, version: int | None = None) -> PromptTemplate:
        versions = self._templates.get(prompt_id)
        if not versions:
            raise PromptError(f"unknown prompt {prompt_id!r}")
        if version is None:
            return versions[max(versions)]
        if version not in versions:
            raise PromptError(f"unknown prompt version {prompt_id}@v{version}")
        return versions[version]

    def ids(self) -> list[str]:
        return sorted(self._templates)
