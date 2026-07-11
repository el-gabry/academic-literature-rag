from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, field_validator, model_validator


class SemanticSearchResult(BaseModel):
    """A ranked text chunk returned by semantic search."""

    text_chunk_id: UUID
    pdf_asset_id: UUID

    chunk_index: int

    start_page_number: int
    end_page_number: int

    text: str

    similarity_score: float
    embedding_model: str

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(
        cls,
        value: int,
    ) -> int:
        """Reject invalid chunk indexes."""

        if value < 0:
            raise ValueError("Semantic search result chunk index must be zero or greater.")

        return value

    @field_validator("start_page_number", "end_page_number")
    @classmethod
    def validate_page_number(
        cls,
        value: int,
    ) -> int:
        """Reject invalid page numbers."""

        if value < 1:
            raise ValueError("Semantic search result page number must be at least 1.")

        return value

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str,
    ) -> str:
        """Reject empty retrieved text."""

        normalized_text = value.strip()

        if not normalized_text:
            raise ValueError("Semantic search result text cannot be empty.")

        return normalized_text

    @field_validator("embedding_model")
    @classmethod
    def validate_embedding_model(
        cls,
        value: str,
    ) -> str:
        """Reject empty embedding model names."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("Semantic search result embedding model cannot be empty.")

        return normalized_value

    @model_validator(mode="after")
    def validate_page_range(
        self,
    ) -> SemanticSearchResult:
        """Reject invalid page ranges."""

        if self.end_page_number < self.start_page_number:
            raise ValueError("Semantic search result end page cannot be before start page.")

        return self
