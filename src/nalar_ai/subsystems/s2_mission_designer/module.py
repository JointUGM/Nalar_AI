from pathlib import Path

from nalar_ai.platform.llm.gateway import CallPolicy
from nalar_ai.settings import Settings
from nalar_ai.shared.question_guard import GuardLexicon
from nalar_ai.subsystems.s2_mission_designer.application.policy import DesignerPolicy as S2Module
from nalar_ai.subsystems.s2_mission_designer.infrastructure.config_loader import load_config

PROMPTS_DIR = Path(__file__).parent / "prompts"
__all__ = ["PROMPTS_DIR", "S2Module", "build_s2_module"]


def build_s2_module(settings: Settings) -> S2Module:
    config = load_config(Path(__file__).parent / "config" / "designer.yaml")
    return S2Module(
        generation_policy=CallPolicy(
            timeout_s=settings.s2_generate_timeout_seconds,
            max_attempts=1,
            max_repairs=1,
            lane="batch",
        ),
        critic_policy=CallPolicy(
            timeout_s=settings.s2_critic_timeout_seconds,
            max_attempts=1,
            max_repairs=1,
            lane="batch",
        ),
        token_budget=settings.s2_grounding_tokens,
        per_target=settings.s2_paragraphs_per_target,
        question_guard=GuardLexicon.build(
            verdict_terms=config.verdict_terms,
            max_chars=config.question_max_chars,
            max_sentences=config.question_max_sentences,
        ),
        anchor_guard=GuardLexicon.build(
            verdict_terms=config.verdict_terms,
            max_chars=config.anchor_max_chars,
            max_sentences=config.anchor_max_sentences,
        ),
    )
