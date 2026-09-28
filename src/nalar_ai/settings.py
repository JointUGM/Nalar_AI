from functools import lru_cache
from typing import Literal

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
    sumopod_messages_base_url: str = ""
    sumopod_openai_base_url: str = ""
    structured_output: Literal["native", "prompt"] = "native"

    model_fast: str = "claude-haiku-4-5"
    model_quality: str = "claude-sonnet-5"
    model_judge: str = "claude-opus-5"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    llm_timeout_seconds: float = 120.0
    llm_max_attempts: int = 3
    llm_concurrency: int = 4
    llm_live_concurrency: int = 32
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


@lru_cache
def get_settings() -> Settings:
    return Settings()  # service_key is read from NALAR_AI_SERVICE_KEY
