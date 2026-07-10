from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from academic_literature_rag.models.chunk_embedding import ChunkEmbedding
from academic_literature_rag.models.text_chunk import TextChunk
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from academic_literature_rag.services.embedding_client import EmbeddingClient


class ChunkEmbeddingError(RuntimeError):
    """Raised when chunk embedding generation fails."""


@dataclass(frozen=True)
class ChunkEmbeddingResult:
    """Summary of one chunk embedding generation attempt."""

    text_chunk_id: UUID
    status: str
    embedding_model: str
    error_message: str | None = None


class ChunkEmbeddingService:
    """Generates and stores embeddings for text chunks."""

    def __init__(
        self,
        *,
        chunk_embedding_repository: ChunkEmbeddingRepository,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._chunk_embedding_repository = chunk_embedding_repository
        self._embedding_client = embedding_client

    def embed_missing_chunks(
        self,
        *,
        limit: int | None = None,
    ) -> list[ChunkEmbeddingResult]:
        """Generate embeddings for chunks missing the configured model."""

        missing_chunks = self._chunk_embedding_repository.list_text_chunks_without_embedding(
            embedding_model=self._embedding_client.model_name,
            limit=limit,
        )

        return [self._embed_one_chunk(text_chunk) for text_chunk in missing_chunks]

    def embed_text_chunk(
        self,
        text_chunk: TextChunk,
    ) -> ChunkEmbedding:
        """Generate and persist one embedding for one text chunk."""

        embedding_response = self._embedding_client.embed_text(
            text_chunk.text,
        )

        if embedding_response.model != self._embedding_client.model_name:
            raise ChunkEmbeddingError("Embedding response model does not match client model.")

        embedding = ChunkEmbedding(
            text_chunk_id=text_chunk.text_chunk_id,
            embedding_model=embedding_response.model,
            embedding_vector=embedding_response.vector,
            embedding_dimension=embedding_response.dimension,
        )

        return self._chunk_embedding_repository.create_or_replace(
            embedding,
        )

    def _embed_one_chunk(
        self,
        text_chunk: TextChunk,
    ) -> ChunkEmbeddingResult:
        """Embed one chunk and convert success or failure into a result."""

        try:
            embedding = self.embed_text_chunk(text_chunk)

            return ChunkEmbeddingResult(
                text_chunk_id=text_chunk.text_chunk_id,
                status="embedded",
                embedding_model=embedding.embedding_model,
            )

        except Exception as error:
            return ChunkEmbeddingResult(
                text_chunk_id=text_chunk.text_chunk_id,
                status="failed",
                embedding_model=self._embedding_client.model_name,
                error_message=f"{type(error).__name__}: {error}",
            )
