from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class PdfPageText(BaseModel):
    """Extracted text from one page of a downloaded PDF."""

    pdf_page_text_id: UUID = Field(default_factory=uuid4)

    pdf_asset_id: UUID

    page_number: int
    text: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("page_number")
    @classmethod
    def validate_page_number(
        cls,
        value: int,
    ) -> int:
        """Reject invalid page numbers."""

        if value < 1:
            raise ValueError("PDF page number must be at least 1.")

        return value

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str,
    ) -> str:
        """Normalize extracted page text."""

        return value.strip()
