from dataclasses import replace

from nalar_ai.subsystems.s5_insight_synthesizer.domain.counts import check_counts
from tests.support.s5 import C_GESEK, M_BERAT, M_HABIS, make_counts


def test_consistent_counts_pass() -> None:
    assert check_counts(make_counts()) == []


def test_aliases_follow_the_request_order() -> None:
    counts = make_counts()
    assert counts.concept_aliases().resolve("c1") == C_GESEK
    assert counts.misconception_aliases().resolve("m2") == M_BERAT
    assert counts.held_counts()[M_HABIS] == 10


def test_names_cover_the_title_concepts_and_statements() -> None:
    names = make_counts().names()
    assert "Gaya dan Gerak" in names and "Kelembaman" in names
    assert "Gaya bisa habis seperti bensin" in names


def test_a_concept_whose_counts_exceed_the_denominator_is_rejected() -> None:
    counts = make_counts()
    first = replace(counts.concepts[0], mastered=10)
    problems = check_counts(replace(counts, concepts=(first, counts.concepts[1])))
    assert problems == [f"concept {C_GESEK}: counts add up to more than the denominator"]


def test_a_resolved_count_above_the_denominator_is_rejected() -> None:
    counts = make_counts()
    first = counts.concepts[0]
    bad = replace(first, misconceptions=(replace(first.misconceptions[0], resolved_count=29),))
    assert check_counts(replace(counts, concepts=(bad, counts.concepts[1]))) == [
        f"misconception {M_HABIS}: a count is above the denominator"
    ]


def test_repeated_ids_are_rejected() -> None:
    counts = make_counts()
    repeated = replace(counts, concepts=(counts.concepts[0], counts.concepts[0]))
    assert "concept ids repeat" in check_counts(repeated)
    assert "misconception ids repeat" in check_counts(repeated)
