from typing import Any

from pydantic import BaseModel

# Claude structured outputs reject these; Pydantic still enforces them client-side.
_UNSUPPORTED = frozenset(
    {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "pattern",
        "default",
        "title",
    }
)
_NAME_MAPPINGS = frozenset({"properties", "$defs"})


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema → the subset Claude structured outputs accept (additionalProperties false)."""
    result = _strip(model.model_json_schema())
    assert isinstance(result, dict)
    return result


def _strip(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _NAME_MAPPINGS and isinstance(value, dict):
            out[key] = {name: _strip(sub) for name, sub in value.items()}
        elif key not in _UNSUPPORTED:
            out[key] = _strip(value)
    if out.get("type") == "object":
        out["additionalProperties"] = False
    return out
