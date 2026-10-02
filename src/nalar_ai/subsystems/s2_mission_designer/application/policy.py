from dataclasses import dataclass

from nalar_ai.platform.llm.gateway import CallPolicy
from nalar_ai.shared.question_guard import GuardLexicon


@dataclass(frozen=True, slots=True)
class DesignerPolicy:
    generation_policy: CallPolicy
    critic_policy: CallPolicy
    token_budget: int
    per_target: int
    question_guard: GuardLexicon
    anchor_guard: GuardLexicon
