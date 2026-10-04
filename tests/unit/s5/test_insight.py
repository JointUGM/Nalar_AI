from collections.abc import Sequence
from typing import Any

import pytest

from nalar_ai.subsystems.s5_insight_synthesizer.domain.insight import (
    Cluster,
    DraftCluster,
    Insight,
    check_insight,
)
from tests.support.s5 import (
    C_GESEK,
    CLUSTERS,
    LEXICON,
    LIMITS,
    M_BERAT,
    M_HABIS,
    NARRATIVE,
    make_counts,
)


def _check(
    narrative: str = NARRATIVE,
    clusters: Sequence[dict[str, Any]] = CLUSTERS,
    suggestions: Sequence[str] = (),
) -> tuple[Insight | None, list[str]]:
    drafts = [
        DraftCluster(c["name"], c["explanation"], tuple(c["misconceptions"])) for c in clusters
    ]
    return check_insight(narrative, drafts, make_counts(), LEXICON, LIMITS, suggestions)


def test_a_clean_insight_is_rewritten_to_ids() -> None:
    insight, problems = _check()
    assert problems == [] and insight is not None
    assert insight.narrative.startswith(f"{{{{count:{M_HABIS}}}}} dari {{{{total}}}} siswa")
    assert insight.clusters == (
        Cluster(
            "Gaya dianggap seperti bahan bakar",
            f"Siswa membayangkan gaya tersimpan di dalam benda; {{{{count:{M_HABIS}}}}} siswa "
            "memegang ide ini.",
            (M_HABIS,),
        ),
        Cluster(
            "Berat menentukan kecepatan jatuh",
            "Siswa mengaitkan berat dengan kecepatan jatuh.",
            (M_BERAT,),
        ),
    )


def test_a_written_number_is_rejected_where_it_appears() -> None:
    _, problems = _check(narrative="Sebagian besar siswa, yaitu 12 anak, masih bingung.")
    assert problems == [
        "narrative: write no digits",
        "narrative: do not say how many students with words (found 'sebagian besar'); "
        "use a placeholder",
    ]


def test_a_number_in_a_cluster_name_is_rejected() -> None:
    clusters = [{**CLUSTERS[0], "name": "Dua ide tentang gaya"}]
    assert _check(clusters=clusters)[1] == ["cluster 1 name: write no number words (found 'dua')"]


def test_a_placeholder_in_a_cluster_name_is_rejected() -> None:
    clusters = [{**CLUSTERS[0], "name": "{{count:m1}} siswa"}]
    assert "cluster 1 name: no placeholders in a name" in _check(clusters=clusters)[1]


def test_empty_clusters_are_valid_when_nobody_holds_an_idea() -> None:
    insight, problems = _check(clusters=[])
    assert problems == [] and insight is not None and insight.clusters == ()


def test_a_cluster_needs_known_held_unused_ideas() -> None:
    clusters = [
        {**CLUSTERS[0], "misconceptions": ["m1", "m3", "m9"]},
        {**CLUSTERS[1], "misconceptions": ["m1"]},
        {**CLUSTERS[1], "misconceptions": []},
    ]
    assert _check(clusters=clusters)[1] == [
        "cluster 1: no student holds m3 now; leave it out",
        "cluster 1: unknown idea 'm9'",
        "cluster 2: m1 is already in another cluster",
        "cluster 3: misconceptions must contain a currently-held idea alias; "
        "otherwise omit this entire cluster",
    ]


def test_word_limits_and_empty_text() -> None:
    assert _check(narrative="kata " * 121)[1] == ["narrative: use at most 120 words"]
    assert _check(narrative="  ")[1] == ["narrative: must not be empty"]


def test_too_many_clusters() -> None:
    clusters = [{**CLUSTERS[1], "misconceptions": []}] * 7
    assert "use at most 6 clusters" in _check(clusters=clusters)[1]


def test_a_teacher_name_with_a_digit_is_not_a_count() -> None:
    counts = make_counts(mission_title="Gaya 2")
    insight, problems = check_insight(
        "Pada misi Gaya 2, {{count:m1}} siswa berpikir gaya bisa habis.",
        [],
        counts,
        LEXICON,
        LIMITS,
        [],
    )
    assert problems == [] and insight is not None


def test_suggestions_are_trimmed_and_all_placeholder_kinds_are_rewritten() -> None:
    text = (
        "  Ajak {{class:total}} siswa menjelaskan gaya gesek; {{class:incomplete}} belum selesai. "
        "Bahas {{count:m1}} dan {{resolved:m1}} siswa, lalu {{mastered:c1}}, "
        "{{developing:c1}} dan {{not_observed:c1}} siswa.  "
    )
    insight, problems = _check(suggestions=[text])
    assert problems == [] and insight is not None
    assert insight.suggestions == (
        "Ajak {{total}} siswa menjelaskan gaya gesek; {{incomplete}} belum selesai. "
        f"Bahas {{{{count:{M_HABIS}}}}} dan {{{{resolved:{M_HABIS}}}}} siswa, lalu "
        f"{{{{mastered:{C_GESEK}}}}}, {{{{developing:{C_GESEK}}}}} dan "
        f"{{{{not_observed:{C_GESEK}}}}} siswa.",
    )


@pytest.mark.parametrize(
    "suggestions,problem",
    [
        (["Ajak 12 siswa menjelaskan."], "write no digits"),
        (["Ajak dua siswa menjelaskan."], "write no number words"),
        (["Ajak sebagian besar siswa menjelaskan."], "do not say how many"),
        (["Bahas {{count:m99}} siswa."], "unknown"),
        (["Bahas {{mastered:c99}} siswa."], "unknown"),
        (["Bahas {{total siswa."], "placeholder"),
        (["  "], "must not be empty"),
        (["Bahas gaya.", " BAHAS   gaya! "], "duplicate"),
        (["kata " * 61], "at most 60 words"),
        (["k" * 601], "at most 600 characters"),
        (["gaya", "gerak", "gesekan", "kelembaman"], "at most 3 suggestions"),
    ],
)
def test_invalid_suggestions_reject_the_whole_insight(suggestions: list[str], problem: str) -> None:
    insight, problems = _check(suggestions=suggestions)
    assert insight is None
    assert any(problem in p for p in problems)


def test_suggestion_character_limit_applies_after_alias_expansion() -> None:
    insight, problems = _check(suggestions=["k" * 565 + " {{count:m1}}"])
    assert insight is None
    assert problems == ["suggestion 1: use at most 600 characters"]


def test_suggestions_at_the_limits_and_empty_list_are_valid() -> None:
    for suggestions in ([], ["kata " * 60, "k" * 600, "Diskusikan gesekan."]):
        insight, problems = _check(suggestions=suggestions)
        assert problems == [] and insight is not None
        assert insight.suggestions == tuple(s.strip() for s in suggestions)
