from nalar_ai.subsystems.s1_knowledge_base.domain.graph import find_cycle


def test_acyclic_graph_has_no_cycle() -> None:
    assert find_cycle([("n2", "n1"), ("n3", "n2"), ("n3", "k1")]) is None


def test_cycle_is_reported_as_a_path() -> None:
    assert find_cycle([("n1", "n2"), ("n2", "n3"), ("n3", "n1")]) == ["n1", "n2", "n3", "n1"]


def test_self_loop_is_a_cycle() -> None:
    assert find_cycle([("n1", "n1")]) == ["n1", "n1"]
