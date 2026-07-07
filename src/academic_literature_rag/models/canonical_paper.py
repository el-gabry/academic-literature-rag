from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class CanonicalPaper(BaseModel):
    """Internal unified representation of one academic paper."""

    canonical_paper_id: UUID = Field(default_factory=uuid4)

    title: str
    normalized_title: str

    doi: str | None = None
    arxiv_id: str | None = None

    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("title", "normalized_title")
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        """Reject blank canonical-paper text fields."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Canonical paper text fields must not be blank.")

        return cleaned_value
