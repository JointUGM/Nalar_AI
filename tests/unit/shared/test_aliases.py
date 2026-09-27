import uuid

import pytest

from nalar_ai.shared.aliases import AliasMap


def test_aliases_are_short_sequential_and_reversible() -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    aliases = AliasMap("c", ids)
    assert aliases.aliases() == ["c1", "c2"]
    assert aliases.alias(ids[1]) == "c2"
    assert aliases.resolve("c1") == ids[0]
    assert "c2" in aliases and "c3" not in aliases
    assert len(aliases) == 2


def test_duplicate_values_share_one_alias() -> None:
    value = uuid.uuid4()
    assert AliasMap("k", [value, value]).aliases() == ["k1"]


def test_alias_resolution_strips_whitespace() -> None:
    value = uuid.uuid4()
    aliases = AliasMap("c", [value])
    assert " c1 " in aliases
    assert aliases.resolve(" c1 ") == value
    assert str(value) not in aliases
    with pytest.raises(KeyError):
        aliases.resolve("c9")
