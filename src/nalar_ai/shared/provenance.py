"""AI provenance (AI-6): one InvocationRecord per model or embedding call, kept per request."""

from dataclasses import dataclass
from uuid import UUID

from nalar_ai.shared.enums import CallStatus, Purpose, RetrievalPath, RetrievalSource
from nalar_ai.shared.errors import BudgetExceededError


@dataclass(frozen=True, slots=True)
class RetrievalRef:
    """One retrieved row that entered a prompt. Maps to ai_invocations.retrieval[] (migration 003)."""

    source: RetrievalSource
    id: UUID
    path: RetrievalPath
    rank: int
    score: float | None = None


@dataclass(frozen=True, slots=True)
class InvocationRecord:
    """Maps 1:1 onto an ai_invocations row, which the backend writes."""

    purpose: Purpose
    model: str
    prompt_version: str
    status: CallStatus
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: int
    cost_usd: float
    request_id: str
    provider: str = "sumopod"
    error_message: str | None = None
    retrieval: tuple[RetrievalRef, ...] = ()


class UsageLedger:
    """Collects every invocation of one HTTP request and enforces its cost cap."""

    def __init__(self, request_id: str, max_cost_usd: float) -> None:
        self.request_id = request_id
        self._max_cost_usd = max_cost_usd
        self._records: list[InvocationRecord] = []

    @property
    def records(self) -> tuple[InvocationRecord, ...]:
        return tuple(self._records)

    @property
    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self._records)

    def add(self, record: InvocationRecord) -> None:
        self._records.append(record)

    def ensure_budget(self) -> None:
        spent = self.total_cost_usd
        if spent >= self._max_cost_usd:
            raise BudgetExceededError(
                "request cost cap reached; refusing further model calls",
                details={"spent_usd": round(spent, 6), "limit_usd": self._max_cost_usd},
            )
