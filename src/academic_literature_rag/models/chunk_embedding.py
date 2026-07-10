from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkEmbedding(BaseModel):
    """Embedding vector generated for one text chunk."""

    chunk_embedding_id: UUID = Field(default_factory=uuid4)

    text_chunk_id: UUID

    embedding_model: str
    embedding_vector: list[float]
    embedding_dimension: int

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(
        cls,
        value: str,
    ) -> str:
        """Reject empty embedding model names."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("Embedding model cannot be empty.")

        return normalized_value

    @field_validator("embedding_vector")
    @classmethod
    def validate_embedding_vector(
        cls,
        value: list[float],
    ) -> list[float]:
        """Reject empty embedding vectors."""

        if not value:
            raise ValueError("Embedding vector cannot be empty.")

        return value

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(
        cls,
        value: int,
    ) -> int:
        """Reject invalid embedding dimensions."""

        if value < 1:
            raise ValueError("Embedding dimension must be positive.")

        return value

    @model_validator(mode="after")
    def validate_vector_dimension(
        self,
    ) -> ChunkEmbedding:
        """Ensure declared dimension matches the vector length."""

        if self.embedding_dimension != len(self.embedding_vector):
            raise ValueError("Embedding dimension must match embedding vector length.")

        return self
