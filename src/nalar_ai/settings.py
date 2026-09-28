from functools import lru_cache
from typing import Any, Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to NALAR_AI_<FIELD_NAME> (see .env.example)."""

    model_config = SettingsConfigDict(
        env_prefix="NALAR_AI_", env_file=".env", extra="ignore", protected_namespaces=()
    )

    environment: str = "local"
    service_key: SecretStr

    llm_provider: Literal["sumopod", "fake"] = "sumopod"
    sumopod_api_key: SecretStr = SecretStr("")
    # chat: /chat/completions (works); messages: /anthropic (404s today, DECISIONS P2)
    sumopod_api: Literal["chat", "messages"] = "chat"
    sumopod_messages_base_url: str = ""
    sumopod_openai_base_url: str = ""
    structured_output: Literal["native", "prompt"] = "native"

    # Tiers from the 2026-09-28 benchmark (DECISIONS M3-M6). Quality stays on Sonnet 5 until a
    # cheaper scorer passes the S4 gate on hand-scored sessions.
    model_fast: str = "gpt-5.4-mini"
    model_quality: str = "claude-sonnet-5"
    model_judge: str = "qwen3.8-max"
    # Extra request fields per model, e.g. switching thinking off where a forced tool call
    # needs it (DECISIONS M1). JSON in NALAR_AI_LLM_MODEL_PARAMS.
    llm_model_params: dict[str, dict[str, Any]] = {"qwen3.8-max": {"enable_thinking": False}}
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    llm_timeout_seconds: float = 120.0
    llm_max_attempts: int = 3
    llm_concurrency: int = 4
    llm_live_concurrency: int = 32
    llm_scoring_concurrency: int = 16
    max_cost_usd_per_request: float = 1.0
    max_upload_bytes: int = 50 * 1024 * 1024

    # S1 policy (design doc §7, §9). Starting points, tuned on Gaya dan Gerak.
    s1_chunk_max_tokens: int = 500
    s1_chunk_min_tail_tokens: int = 80
    s1_chunk_hard_max_tokens: int = 600
    s1_dedup_link_threshold: float = 0.92
    s1_dedup_judge_threshold: float = 0.82
    s1_cp_min_similarity: float = 0.25
    s1_max_section_claude_tokens: int = 80_000
    s1_evidence_pack_claude_tokens: int = 3_000

    # S3 policy (design doc s3 §8, §11, §13). Starting points, tuned on the pilot.
    s3_turn_budget_seconds: float = 4.5
    s3_classify_timeout_seconds: float = 2.0
    s3_choose_timeout_seconds: float = 2.5
    s3_embed_timeout_seconds: float = 1.0
    s3_min_step_seconds: float = 0.3
    s3_min_probes: int = 4
    s3_transcript_turns: int = 6
    s3_max_answer_chars: int = 1500
    s3_guard_max_reference_similarity: float = 0.80
    s3_guard_min_approved_similarity: float = 0.55

    # S4 policy (design doc s4 §7, §11). Starting points, tuned on the pilot.
    s4_score_timeout_seconds: float = 60.0
    s4_reflect_timeout_seconds: float = 20.0
    s4_max_attempts: int = 2
    s4_max_answer_chars: int = 1500
    s4_max_quotes_per_score: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()  # service_key is read from NALAR_AI_SERVICE_KEY
