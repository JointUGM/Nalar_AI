"""Count placeholders (design doc §4.2). The model writes aliases; code rewrites them to ids.

The model writes {{class:total}} rather than {{total}}: the prompt registry reserves {{name}}.
"""

import re
from uuid import UUID

from nalar_ai.shared.aliases import AliasMap

_TOKEN = re.compile(r"\{\{([^{}]*)\}\}")
_CLASS_KINDS = frozenset({"total", "incomplete"})
_CONCEPT_KINDS = frozenset({"mastered", "developing", "not_observed"})
_MISCONCEPTION_KINDS = frozenset({"count", "resolved"})


def strip_placeholders(text: str) -> str:
    return _TOKEN.sub(" ", text)


def has_braces(text: str) -> bool:
    return "{" in text or "}" in text


def rewrite(
    text: str, concepts: AliasMap[UUID], misconceptions: AliasMap[UUID]
) -> tuple[str, list[str]]:
    """The text with backend placeholders, and the problems found (empty means valid)."""
    problems: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kind, _, alias = (part.strip().casefold() for part in match.group(1).partition(":"))
        if kind == "class" and alias in _CLASS_KINDS:
            return f"{{{{{alias}}}}}"
        table = (
            concepts
            if kind in _CONCEPT_KINDS
            else misconceptions
            if kind in _MISCONCEPTION_KINDS
            else None
        )
        if table is None or alias not in table:
            problems.append(f"unknown placeholder '{match.group(0)[:40]}'")
            return match.group(0)
        return f"{{{{{kind}:{table.resolve(alias)}}}}}"

    rewritten = _TOKEN.sub(replace, text)
    if has_braces(strip_placeholders(text)):
        problems.append("write placeholders exactly as {{kind:id}}")
    return rewritten, problems
