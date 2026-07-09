from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class TextChunk(BaseModel):
    """A clean text chunk ready for embedding and retrieval."""

    text_chunk_id: UUID = Field(default_factory=uuid4)

    pdf_asset_id: UUID

    chunk_index: int

    start_page_number: int
    end_page_number: int

    text: str
    char_count: int

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(
        cls,
        value: int,
    ) -> int:
        """Reject invalid chunk indexes."""

        if value < 0:
            raise ValueError("Text chunk index must be zero or greater.")

        return value

    @field_validator("start_page_number", "end_page_number")
    @classmethod
    def validate_page_number(
        cls,
        value: int,
    ) -> int:
        """Reject invalid page numbers."""

        if value < 1:
            raise ValueError("Text chunk page number must be at least 1.")

        return value

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str,
    ) -> str:
        """Normalize and reject empty chunk text."""

        normalized_text = value.strip()

        if not normalized_text:
            raise ValueError("Text chunk cannot be empty.")

        return normalized_text

    @field_validator("char_count")
    @classmethod
    def validate_char_count(
        cls,
        value: int,
    ) -> int:
        """Reject invalid character counts."""

        if value < 1:
            raise ValueError("Text chunk character count must be positive.")

        return value

    @model_validator(mode="after")
    def validate_page_range(
        self,
    ) -> TextChunk:
        """Reject invalid page ranges."""

        if self.end_page_number < self.start_page_number:
            raise ValueError("Text chunk end page cannot be before start page.")

        if self.char_count != len(self.text):
            raise ValueError("Text chunk character count must match text length.")

        return self
