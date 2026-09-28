"""Step 2: code checks every scoring answer before it can be returned (design doc §7).

The model speaks in aliases (t0 turns, c1 concepts, m1 misconceptions). Nothing is resolved
to an id until every rule passes, so an invalid answer never produces a partial result.
"""

import re
from dataclasses import dataclass
from uuid import UUID

from nalar_ai.shared.aliases import AliasMap
from nalar_ai.shared.enums import ConceptOutcome, RubricDimension
from nalar_ai.subsystems.s4_session_evaluator.domain.quotes import find_quote
from nalar_ai.subsystems.s4_session_evaluator.domain.session import EvalSession, EvalTurn

_DIGIT = re.compile(r"\d")
MAX_LEVEL = 4


@dataclass(frozen=True, slots=True)
class VerificationLimits:
    max_quotes_per_score: int = 4
    max_quote_chars: int = 300
    max_rationale_chars: int = 400
    max_summary_chars: int = 600


# ---- what the model returned (aliases, unchecked) ----


@dataclass(frozen=True, slots=True)
class DraftEvidence:
    turn: str
    quote: str


@dataclass(frozen=True, slots=True)
class DraftScore:
    dimension: RubricDimension
    evidence: tuple[DraftEvidence, ...]
    rationale: str
    level: int


@dataclass(frozen=True, slots=True)
class DraftConcept:
    concept: str
    evidence_turn: str | None
    initial_misconception: str | None
    outcome: ConceptOutcome
    misconception: str | None


@dataclass(frozen=True, slots=True)
class DraftTurnQuality:
    turn: str
    quality: int


@dataclass(frozen=True, slots=True)
class ScoringDraft:
    scores: tuple[DraftScore, ...]
    concepts: tuple[DraftConcept, ...]
    turn_quality: tuple[DraftTurnQuality, ...]
    summary: str


# ---- the verified result (ids) ----


@dataclass(frozen=True, slots=True)
class Evidence:
    turn_id: UUID
    quote: str  # the span of the student's answer, never the model's rewording


@dataclass(frozen=True, slots=True)
class Score:
    dimension: RubricDimension
    level: int
    rationale: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class ConceptResult:
    concept_id: UUID
    outcome: ConceptOutcome
    misconception_id: UUID | None
    evidence_turn_id: UUID | None
    initial_misconception_id: UUID | None
    resolved_in_session: bool


@dataclass(frozen=True, slots=True)
class TurnQuality:
    turn_id: UUID
    turn_index: int
    quality: int


@dataclass(frozen=True, slots=True)
class Evaluation:
    summary: str
    scores: tuple[Score, ...]  # rubric order
    concept_results: tuple[ConceptResult, ...]  # target order
    turn_quality: tuple[TurnQuality, ...]  # turn order


@dataclass(frozen=True, slots=True)
class SessionAliases:
    turns: dict[str, EvalTurn]  # answered turns only
    concepts: AliasMap[UUID]
    misconceptions: AliasMap[UUID]
    misconception_concept: dict[UUID, UUID]

    @classmethod
    def of(cls, session: EvalSession) -> "SessionAliases":
        pack = session.pack
        return cls(
            turns={turn.alias: turn for turn in session.answered_turns},
            concepts=AliasMap("c", [target.id for target in pack.targets]),
            misconceptions=AliasMap("m", [m.id for m in pack.misconceptions]),
            misconception_concept={m.id: m.concept_id for m in pack.misconceptions},
        )


def verify(
    draft: ScoringDraft, session: EvalSession, limits: VerificationLimits
) -> tuple[Evaluation | None, list[str]]:
    """(evaluation, []) when every rule holds, otherwise (None, problems for the repair call)."""
    aliases = SessionAliases.of(session)
    errors: list[str] = []
    scores = _scores(draft, aliases, limits, errors)
    concepts = _concepts(draft, aliases, errors)
    quality = _turn_quality(draft, aliases, errors)
    summary = draft.summary.strip()
    if not summary or len(summary) > limits.max_summary_chars:
        errors.append(f"summary: must be 1 to {limits.max_summary_chars} characters")
    if _DIGIT.search(summary):
        errors.append("summary: write no digits; counts are never written by the model")
    if errors:
        return None, errors
    return Evaluation(summary, scores, concepts, quality), []


def _scores(
    draft: ScoringDraft, aliases: SessionAliases, limits: VerificationLimits, errors: list[str]
) -> tuple[Score, ...]:
    dimensions = [score.dimension for score in draft.scores]
    if sorted(dimensions) != sorted(RubricDimension):
        errors.append("scores: list claim, evidence, mechanism and transfer exactly once each")
    verified: dict[RubricDimension, Score] = {}
    for score in draft.scores:
        where = f"scores[{score.dimension.value}]"
        if not 0 <= score.level <= MAX_LEVEL:
            errors.append(f"{where}.level: must be 0 to {MAX_LEVEL}")
        rationale = score.rationale.strip()
        if not rationale or len(rationale) > limits.max_rationale_chars:
            errors.append(
                f"{where}.rationale: must be 1 to {limits.max_rationale_chars} characters"
            )
        if not 1 <= len(score.evidence) <= limits.max_quotes_per_score:
            errors.append(
                f"{where}.evidence: give 1 to {limits.max_quotes_per_score} quotes; for a level 0 "
                "quote the student's closest attempt"
            )
        evidence: list[Evidence] = []
        for number, item in enumerate(score.evidence):
            span = _quote(item, f"{where}.evidence[{number}]", aliases, limits, errors)
            if span is not None:
                evidence.append(span)
        verified[score.dimension] = Score(score.dimension, score.level, rationale, tuple(evidence))
    return tuple(verified[d] for d in RubricDimension if d in verified)


def _quote(
    item: DraftEvidence,
    where: str,
    aliases: SessionAliases,
    limits: VerificationLimits,
    errors: list[str],
) -> Evidence | None:
    turn = aliases.turns.get(item.turn.strip())
    if turn is None:
        errors.append(f"{where}.turn: {item.turn!r} is not an answered turn")
        return None
    if len(item.quote) > limits.max_quote_chars:
        errors.append(f"{where}.quote: at most {limits.max_quote_chars} characters")
        return None
    span = find_quote(turn.answer_text, item.quote)
    if span is None:
        errors.append(
            f"{where}.quote: not found in the student's answer in {turn.alias}; copy the "
            "student's words exactly, never the question's"
        )
        return None
    return Evidence(turn.turn_id, span)


def _concepts(
    draft: ScoringDraft, aliases: SessionAliases, errors: list[str]
) -> tuple[ConceptResult, ...]:
    expected = aliases.concepts.aliases()
    given = [concept.concept.strip() for concept in draft.concepts]
    if sorted(given) != sorted(expected):
        errors.append(f"concepts: give exactly one entry for each of {', '.join(expected)}")
    verified: dict[UUID, ConceptResult] = {}
    for item in draft.concepts:
        where = f"concepts[{item.concept}]"
        if item.concept not in aliases.concepts:
            continue  # reported above
        concept_id = aliases.concepts.resolve(item.concept)
        misconception = _misconception(item.misconception, concept_id, where, aliases, errors)
        initial = _misconception(
            item.initial_misconception, concept_id, f"{where}.initial", aliases, errors
        )
        is_misconception = item.outcome is ConceptOutcome.MISCONCEPTION
        if is_misconception != (item.misconception is not None):
            errors.append(
                f"{where}: name a misconception exactly when the outcome is 'misconception'"
            )
        if item.outcome is ConceptOutcome.NOT_OBSERVED and item.initial_misconception is not None:
            errors.append(
                f"{where}: a concept with an initial misconception was observed; "
                "do not use 'not_observed'"
            )
        # Computed, never taken from the model: a contradictory flag would skew the class map
        # and the reflection's "where you changed your mind" (design doc §6).
        resolved = initial is not None and not is_misconception
        evidence_turn_id: UUID | None = None
        if item.evidence_turn is not None:
            turn = aliases.turns.get(item.evidence_turn.strip())
            if turn is None:
                errors.append(
                    f"{where}.evidence_turn: {item.evidence_turn!r} is not an answered turn"
                )
            else:
                evidence_turn_id = turn.turn_id
        elif item.outcome is not ConceptOutcome.NOT_OBSERVED:
            errors.append(f"{where}.evidence_turn: required unless the outcome is 'not_observed'")
        verified[concept_id] = ConceptResult(
            concept_id, item.outcome, misconception, evidence_turn_id, initial, resolved
        )
    order = (aliases.concepts.resolve(alias) for alias in expected)
    return tuple(verified[c] for c in order if c in verified)


def _misconception(
    alias: str | None, concept_id: UUID, where: str, aliases: SessionAliases, errors: list[str]
) -> UUID | None:
    if alias is None:
        return None
    if alias not in aliases.misconceptions:
        errors.append(f"{where}.misconception: {alias!r} is not a listed wrong idea")
        return None
    misconception_id = aliases.misconceptions.resolve(alias)
    if aliases.misconception_concept[misconception_id] != concept_id:
        errors.append(f"{where}.misconception: {alias} belongs to another concept")
        return None
    return misconception_id


def _turn_quality(
    draft: ScoringDraft, aliases: SessionAliases, errors: list[str]
) -> tuple[TurnQuality, ...]:
    expected = list(aliases.turns)
    given = [item.turn.strip() for item in draft.turn_quality]
    if sorted(given) != sorted(expected):
        errors.append(f"turn_quality: rate exactly these turns once each: {', '.join(expected)}")
    verified: dict[str, TurnQuality] = {}
    for item in draft.turn_quality:
        turn = aliases.turns.get(item.turn.strip())
        if turn is None:
            continue  # reported above
        if not 0 <= item.quality <= MAX_LEVEL:
            errors.append(f"turn_quality[{turn.alias}]: must be 0 to {MAX_LEVEL}")
        verified[turn.alias] = TurnQuality(turn.turn_id, turn.turn_index, item.quality)
    return tuple(verified[a] for a in expected if a in verified)
