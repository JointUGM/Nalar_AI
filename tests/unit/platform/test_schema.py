from pydantic import BaseModel, Field

from nalar_ai.platform.llm.schema import strict_json_schema


class _Inner(BaseModel):
    pattern: str = Field(min_length=1)


class _Outer(BaseModel):
    key: str = Field(min_length=1, max_length=10)
    items: list[_Inner] = Field(min_length=1)
    library: str | None


def test_strips_unsupported_constraints_but_keeps_property_names() -> None:
    schema = strict_json_schema(_Outer)
    text = str(schema)
    assert "minLength" not in text and "maxLength" not in text and "minItems" not in text
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["_Inner"]["additionalProperties"] is False
    assert "pattern" in schema["$defs"]["_Inner"]["properties"]
    assert schema["required"] == ["key", "items", "library"]
