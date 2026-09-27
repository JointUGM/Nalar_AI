from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel

from nalar_ai.shared.enums import AiPurpose, CallStatus, RetrievalPath, RetrievalSource
from nalar_ai.shared.provenance import InvocationRecord, UsageLedger


class RetrievalRefOut(BaseModel):
    source: RetrievalSource
    id: UUID
    path: RetrievalPath
    rank: int
    score: float | None


class InvocationOut(BaseModel):
    """One ai_invocations row for the backend to insert unchanged (AI-6)."""

    purpose: AiPurpose
    model: str
    prompt_version: str
    provider: str
    status: CallStatus
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: int
    cost_usd: float
    request_id: str
    error_message: str | None
    retrieval: list[RetrievalRefOut]

    @classmethod
    def from_record(cls, record: InvocationRecord) -> "InvocationOut":
        return cls(
            purpose=record.purpose,
            model=record.model,
            prompt_version=record.prompt_version,
            provider=record.provider,
            status=record.status,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_read_tokens=record.cache_read_tokens,
            cache_write_tokens=record.cache_write_tokens,
            latency_ms=record.latency_ms,
            cost_usd=record.cost_usd,
            request_id=record.request_id,
            error_message=record.error_message,
            retrieval=[
                RetrievalRefOut(
                    source=ref.source, id=ref.id, path=ref.path, rank=ref.rank, score=ref.score
                )
                for ref in record.retrieval
            ],
        )


class Envelope[T](BaseModel):
    result: T
    invocations: list[InvocationOut]
    warnings: list[str]


def envelope[T](result: T, ledger: UsageLedger, warnings: Sequence[str] = ()) -> Envelope[T]:
    return Envelope(
        result=result,
        invocations=[InvocationOut.from_record(record) for record in ledger.records],
        warnings=list(warnings),
    )
