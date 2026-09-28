"""Steps 1-2: Sonnet scores the session, code verifies it, one repair if needed (design doc §6, §7)."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.shared.enums import ConceptOutcome, RubricDimension
from nalar_ai.shared.provenance import UsageLedger
from nalar_ai.shared.text import fence_untrusted
from nalar_ai.subsystems.s4_session_evaluator.application.policy import EvaluatorPolicy
from nalar_ai.subsystems.s4_session_evaluator.application.rendering import (
    concepts_block,
    misconceptions_block,
    rubric_block,
    transcript,
)
from nalar_ai.subsystems.s4_session_evaluator.domain.session import EvalSession
from nalar_ai.subsystems.s4_session_evaluator.domain.verification import (
    DraftConcept,
    DraftEvidence,
    DraftScore,
    DraftTurnQuality,
    Evaluation,
    ScoringDraft,
    SessionAliases,
    verify,
)


class EvidenceOut(BaseModel):
    turn: str
    quote: str


class ScoreOut(BaseModel):
    dimension: Literal["claim", "evidence", "mechanism", "transfer"]
    evidence: list[EvidenceOut]  # before level: quotes first (design doc §6)
    rationale: str
    level: int


class ConceptOut(BaseModel):
    concept: str
    evidence_turn: str | None
    initial_misconception: str | None
    outcome: Literal["mastered", "developing", "misconception", "not_observed"]
    misconception: str | None


class TurnQualityOut(BaseModel):
    turn: str
    quality: int


class ScoringOut(BaseModel):
    scores: list[ScoreOut]
    concepts: list[ConceptOut]
    turn_quality: list[TurnQualityOut]
    summary: str

    def to_draft(self) -> ScoringDraft:
        return ScoringDraft(
            scores=tuple(
                DraftScore(
                    dimension=RubricDimension(s.dimension),
                    evidence=tuple(DraftEvidence(e.turn, e.quote) for e in s.evidence),
                    rationale=s.rationale,
                    level=s.level,
                )
                for s in self.scores
            ),
            concepts=tuple(
                DraftConcept(
                    concept=c.concept,
                    evidence_turn=c.evidence_turn,
                    initial_misconception=c.initial_misconception,
                    outcome=ConceptOutcome(c.outcome),
                    misconception=c.misconception,
                )
                for c in self.concepts
            ),
            turn_quality=tuple(DraftTurnQuality(q.turn, q.quality) for q in self.turn_quality),
            summary=self.summary,
        )


@dataclass(frozen=True, slots=True)
class ScoringResult:
    evaluation: Evaluation
    repaired: bool


class ScoreSessionUseCase:
    PROMPT_ID = "s4.score_session"

    def __init__(self, *, llm: LLMGateway, policy: EvaluatorPolicy) -> None:
        self._llm = llm
        self._policy = policy

    async def execute(self, session: EvalSession, ledger: UsageLedger) -> ScoringResult:
        """Raises OutputValidationError (502) when the answer is still invalid after the repair."""
        aliases = SessionAliases.of(session)
        verified: list[Evaluation] = []
        checks = 0

        def check(out: ScoringOut) -> list[str]:
            nonlocal checks
            checks += 1
            evaluation, errors = verify(out.to_draft(), session, self._policy.limits)
            if evaluation is not None:
                verified.append(evaluation)
            return errors

        pack = session.pack
        await self._llm.generate(
            prompt_id=self.PROMPT_ID,
            variables={
                "anchor": fence_untrusted("anchor", pack.anchor_problem),
                "reference": fence_untrusted("reference", pack.reference_reasoning),
                "rubric": fence_untrusted("rubric", rubric_block(session.rubric)),
                "concepts": fence_untrusted(
                    "concepts", concepts_block(pack.targets, aliases.concepts)
                ),
                "misconceptions": fence_untrusted(
                    "wrong_ideas",
                    misconceptions_block(
                        pack.misconceptions, aliases.concepts, aliases.misconceptions
                    ),
                ),
                "transcript": transcript(
                    session.turns, aliases.concepts, max_answer_chars=self._policy.max_answer_chars
                ),
                "answered": ", ".join(aliases.turns),
            },
            output_model=ScoringOut,
            ledger=ledger,
            semantic_check=check,
            policy=self._policy.score_call(),
        )
        return ScoringResult(evaluation=verified[-1], repaired=checks > 1)
