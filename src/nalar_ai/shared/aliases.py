from collections.abc import Hashable, Iterable


class AliasMap[T: Hashable]:
    """Short prompt aliases (c1, k2, L3) for long ids.

    Aliases cost about 2 tokens instead of about 25 for a UUID, and a model can't corrupt
    them into a near-valid id: anything that is not an exact alias is rejected.
    """

    def __init__(self, prefix: str, values: Iterable[T]) -> None:
        self._by_alias: dict[str, T] = {}
        self._by_value: dict[T, str] = {}
        for value in values:
            if value in self._by_value:
                continue
            alias = f"{prefix}{len(self._by_alias) + 1}"
            self._by_alias[alias] = value
            self._by_value[value] = alias

    def alias(self, value: T) -> str:
        return self._by_value[value]

    def resolve(self, alias: str) -> T:
        return self._by_alias[alias.strip()]

    def aliases(self) -> list[str]:
        return list(self._by_alias)

    def __contains__(self, alias: object) -> bool:
        return isinstance(alias, str) and alias.strip() in self._by_alias

    def __len__(self) -> int:
        return len(self._by_alias)
