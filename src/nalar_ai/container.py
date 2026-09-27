from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from nalar_ai.platform.embeddings.fakes import HashingEmbedder
from nalar_ai.platform.embeddings.openai_compat import OpenAICompatEmbedder
from nalar_ai.platform.embeddings.ports import EmbeddingPort
from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.llm.anthropic_adapter import AnthropicMessagesAdapter
from nalar_ai.platform.llm.fakes import ScriptedLLM
from nalar_ai.platform.llm.gateway import LLMGateway
from nalar_ai.platform.llm.ports import LLMPort
from nalar_ai.platform.prompts.registry import PromptRegistry
from nalar_ai.platform.tokens import TokenCounter
from nalar_ai.settings import Settings
from nalar_ai.shared.enums import ModelTier
from nalar_ai.subsystems.s1_knowledge_base.module import PROMPTS_DIR as S1_PROMPTS_DIR
from nalar_ai.subsystems.s1_knowledge_base.module import S1Module, build_s1_module

# Every subsystem registers its prompt directory here.
PROMPT_DIRECTORIES: tuple[Path, ...] = (S1_PROMPTS_DIR,)

Closer = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Container:
    """Composition root: every long-lived object the app needs, built once at startup."""

    settings: Settings
    prompts: PromptRegistry
    llm: LLMGateway
    embeddings: EmbeddingService
    tokens: TokenCounter
    s1: S1Module
    closers: tuple[Closer, ...] = field(default=())

    async def aclose(self) -> None:
        for close in self.closers:
            await close()


def build_container(
    settings: Settings,
    *,
    llm_port: LLMPort | None = None,
    embedding_port: EmbeddingPort | None = None,
) -> Container:
    closers: list[Closer] = []
    if llm_port is None:
        llm_port = _default_llm_port(settings, closers)
    if embedding_port is None:
        embedding_port = _default_embedding_port(settings, closers)
    prompts = PromptRegistry.from_directories(PROMPT_DIRECTORIES)
    llm = LLMGateway(
        port=llm_port,
        prompts=prompts,
        models={
            ModelTier.FAST: settings.model_fast,
            ModelTier.QUALITY: settings.model_quality,
            ModelTier.JUDGE: settings.model_judge,
        },
        native_structured_output=settings.structured_output == "native",
        max_attempts=settings.llm_max_attempts,
        concurrency=settings.llm_concurrency,
        provider=settings.llm_provider,
    )
    embeddings = EmbeddingService(
        embedding_port,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        max_attempts=settings.llm_max_attempts,
        provider=settings.llm_provider,
    )
    return Container(
        settings=settings,
        prompts=prompts,
        llm=llm,
        embeddings=embeddings,
        tokens=TokenCounter(),
        s1=build_s1_module(settings),
        closers=tuple(closers),
    )


def _default_llm_port(settings: Settings, closers: list[Closer]) -> LLMPort:
    if settings.llm_provider == "fake":
        return ScriptedLLM()
    adapter = AnthropicMessagesAdapter.from_settings(settings)
    closers.append(adapter.aclose)
    return adapter


def _default_embedding_port(settings: Settings, closers: list[Closer]) -> EmbeddingPort:
    if settings.llm_provider == "fake":
        return HashingEmbedder()
    embedder = OpenAICompatEmbedder.from_settings(settings)
    closers.append(embedder.aclose)
    return embedder
