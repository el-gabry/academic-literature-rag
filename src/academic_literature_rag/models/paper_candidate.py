from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PaperCandidate(BaseModel):
    """A paper record retrieved from an external source before merging."""

    source: Literal["semantic_scholar", "arxiv"]
    source_id: str
    title: str
    landing_url: str
    retrieved_at: datetime

    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    open_access_pdf_url: str | None = None
    citation_count: int | None = None

    @field_validator("source_id", "title", "landing_url")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Field must not be blank.")

        return cleaned_value
