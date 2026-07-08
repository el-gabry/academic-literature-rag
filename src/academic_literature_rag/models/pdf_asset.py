from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


PdfDownloadStatus = Literal[
    "pending",
    "downloaded",
    "failed",
]


class PdfAsset(BaseModel):
    """Represents one PDF candidate or downloaded PDF file."""

    pdf_asset_id: UUID = Field(default_factory=uuid4)

    canonical_paper_id: UUID
    source_paper_id: UUID | None = None

    source_url: str

    download_status: PdfDownloadStatus = "pending"

    local_file_path: str | None = None
    sha256_checksum: str | None = None
    content_type: str | None = None
    file_size_bytes: int | None = None
    failure_message: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    downloaded_at: datetime | None = None

    @field_validator("source_url")
    @classmethod
    def validate_source_url(
        cls,
        value: str,
    ) -> str:
        """Reject blank PDF source URLs."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("PDF source URL must not be blank.")

        return cleaned_value

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(
        cls,
        value: int | None,
    ) -> int | None:
        """Reject negative file sizes."""

        if value is not None and value < 0:
            raise ValueError("PDF file size must not be negative.")

        return value
