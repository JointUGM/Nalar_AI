from collections import defaultdict
from collections.abc import Iterable


def find_cycle(edges: Iterable[tuple[str, str]]) -> list[str] | None:
    """Edges are (concept, prerequisite). Returns one cycle as a node path, or None."""
    graph: defaultdict[str, list[str]] = defaultdict(list)
    for concept, prerequisite in edges:
        graph[concept].append(prerequisite)
    state: dict[str, int] = {}  # 1 = on the current path, 2 = finished
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for nxt in sorted(graph.get(node, [])):
            if state.get(nxt) == 1:
                return [*path[path.index(nxt) :], nxt]
            if nxt not in state:
                found = visit(nxt)
                if found is not None:
                    return found
        path.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if node not in state:
            found = visit(node)
            if found is not None:
                return found
    return None
