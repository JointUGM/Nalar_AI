from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.prompts.registry import PromptRegistry
from nalar_ai.shared.enums import ModelTier
from nalar_ai.subsystems.s1_knowledge_base.module import PROMPTS_DIR

MODELS = {
    ModelTier.FAST: "claude-haiku-4-5",
    ModelTier.QUALITY: "claude-sonnet-5",
    ModelTier.JUDGE: "claude-opus-5",
}


def make_gateway(llm: ScriptedLLM) -> LLMGateway:
    return LLMGateway(
        port=llm, prompts=PromptRegistry.from_directories([PROMPTS_DIR]), models=MODELS
    )


def make_embeddings() -> EmbeddingService:
    return EmbeddingService(HashingEmbedder(), model="text-embedding-3-small", dimensions=1536)
