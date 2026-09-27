import hashlib
import math
import re
from collections.abc import Sequence

from nalar_ai.platform.embeddings.ports import EmbeddingBatch

_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic bag-of-words vectors: texts that share words are similar.

    For tests, offline evals and the `fake` provider only. Never used for stored embeddings.
    """

    async def embed(self, texts: Sequence[str], *, model: str, dimensions: int) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=[self.vector(text, dimensions) for text in texts],
            total_tokens=sum(len(_WORD.findall(text)) for text in texts),
        )

    @staticmethod
    def vector(text: str, dimensions: int) -> list[float]:
        values = [0.0] * dimensions
        for word in _WORD.findall(text.casefold()):
            digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            values[int.from_bytes(digest, "big") % dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            values[0] = 1.0
            return values
        return [value / norm for value in values]
