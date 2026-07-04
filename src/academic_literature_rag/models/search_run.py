from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRun(BaseModel):
    """Metadata for one paper-retrieval operation."""

    model_config = ConfigDict(validate_assignment=True)

    run_id: UUID = Field(default_factory=uuid4)
    source: Literal["semantic_scholar", "arxiv"]
    query: str

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    status: Literal["running", "completed", "failed"] = "running"
    result_count: int | None = Field(default=None, ge=0)

    raw_response_path: str | None = None
    error_message: str | None = None

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("Search query must not be blank.")

        return cleaned_value
