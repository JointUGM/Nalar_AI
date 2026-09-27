from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, StringConstraints

from nalar_ai.platform.embeddings.service import EmbeddingService
from nalar_ai.platform.http.dependencies import LedgerDep, get_embedding_service
from nalar_ai.platform.http.envelope import Envelope, envelope
from nalar_ai.platform.http.security import require_service_key

router = APIRouter(prefix="/v1", tags=["embeddings"], dependencies=[Depends(require_service_key)])

EmbeddingTag = Literal["query", "chunk", "concept", "misconception", "cp_statement", "library"]
EmbeddingText = Annotated[str, StringConstraints(min_length=1, max_length=30_000)]


class EmbedIn(BaseModel):
    texts: list[EmbeddingText] = Field(min_length=1, max_length=256)
    tag: EmbeddingTag


class EmbedOut(BaseModel):
    vectors: list[list[float]]
    embedding_model: str
    dimensions: int


@router.post("/embeddings", response_model=Envelope[EmbedOut])
async def embed(
    body: EmbedIn,
    ledger: LedgerDep,
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> Envelope[EmbedOut]:
    """Embeddings for seeds (CP statements, library), queries and hand-added items."""
    vectors = await embeddings.embed(body.texts, tag=body.tag, ledger=ledger)
    return envelope(
        EmbedOut(
            vectors=[list(vector) for vector in vectors],
            embedding_model=embeddings.model,
            dimensions=embeddings.dimensions,
        ),
        ledger,
    )
