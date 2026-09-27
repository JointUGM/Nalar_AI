import uuid

from nalar_ai.subsystems.s1_knowledge_base.domain.dedup_policy import (
    Candidate,
    DedupBasis,
    DedupThresholds,
    decide,
)

T = DedupThresholds()


def _c(name: str, similarity: float) -> Candidate:
    return Candidate(uuid.uuid4(), name, "", similarity)


def test_same_name_links_regardless_of_similarity() -> None:
    candidate = _c("Gaya Gesek", 0.4)
    decision = decide("gaya gesek", [candidate], T)
    assert (decision.action, decision.basis, decision.concept_id) == (
        "link",
        DedupBasis.NAME_MATCH,
        candidate.concept_id,
    )


def test_very_close_links_by_similarity() -> None:
    best = _c("Gesekan", 0.95)
    decision = decide("Gaya gesek", [_c("Massa", 0.5), best], T)
    assert (decision.action, decision.basis, decision.concept_id) == (
        "link",
        DedupBasis.SIMILARITY,
        best.concept_id,
    )


def test_grey_zone_goes_to_the_judge_with_at_most_three_candidates() -> None:
    candidates = [_c(f"k{i}", s) for i, s in enumerate([0.83, 0.91, 0.85, 0.88, 0.5])]
    decision = decide("Gaya gesek", candidates, T)
    assert decision.action == "judge"
    assert [c.similarity for c in decision.judge_candidates] == [0.91, 0.88, 0.85]


def test_far_or_missing_candidates_create() -> None:
    assert decide("Gaya gesek", [_c("Massa", 0.6)], T).basis is DedupBasis.SIMILARITY
    assert decide("Gaya gesek", [_c("Massa", 0.6)], T).action == "create"
    assert decide("Gaya gesek", [], T).basis is DedupBasis.NO_CANDIDATE
