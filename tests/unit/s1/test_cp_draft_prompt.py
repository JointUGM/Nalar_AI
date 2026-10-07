from nalar_ai.platform.prompts.registry import PromptRegistry
from nalar_ai.settings import Settings
from nalar_ai.shared.enums import AiPurpose, ModelTier
from nalar_ai.subsystems.s1_knowledge_base.module import PROMPTS_DIR


def test_cp_draft_prompt_is_a_quality_tier_cp_extract_call() -> None:
    template = PromptRegistry.from_directories([PROMPTS_DIR]).get("s1.cp_draft")
    assert template.version_tag == "s1.cp_draft@v1"
    assert template.purpose is AiPurpose.CP_EXTRACT
    assert template.tier is ModelTier.QUALITY


def test_cp_excerpt_limit_is_a_setting() -> None:
    assert Settings(service_key="k").s1_cp_excerpt_max_chars == 60_000
