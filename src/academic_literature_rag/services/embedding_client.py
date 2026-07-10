from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingResponse:
    """Embedding output returned by an embedding provider."""

    model: str
    vector: list[float]

    @property
    def dimension(
        self,
    ) -> int:
        """Return the embedding vector dimension."""

        return len(self.vector)


class EmbeddingClient(Protocol):
    """Interface for embedding providers."""

    @property
    def model_name(
        self,
    ) -> str:
        """Return the embedding model name."""

    def embed_text(
        self,
        text: str,
    ) -> EmbeddingResponse:
        """Generate one embedding vector for one text input."""
