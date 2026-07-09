from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import httpx

from academic_literature_rag.models.pdf_asset import PdfAsset
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetNotFoundError,
    PdfAssetRepository,
)


class PdfDownloadError(RuntimeError):
    """Raised when a PDF asset cannot be downloaded or validated."""


class PdfDownloadService:
    """Downloads and validates PDF assets."""

    def __init__(
        self,
        *,
        pdf_asset_repository: PdfAssetRepository,
        pdf_storage_directory: Path,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._pdf_asset_repository = pdf_asset_repository
        self._pdf_storage_directory = pdf_storage_directory
        self._http_client = http_client or httpx.Client(timeout=30.0)
        self._owns_http_client = http_client is None

    def close(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_http_client:
            self._http_client.close()

    def download(
        self,
        pdf_asset_id: UUID,
    ) -> PdfAsset:
        """Download one PDF asset and update its persisted status."""

        pdf_asset = self._pdf_asset_repository.get(pdf_asset_id)

        if pdf_asset is None:
            raise PdfAssetNotFoundError(f"PDF asset does not exist: {pdf_asset_id}")

        try:
            response = self._http_client.get(
                pdf_asset.source_url,
                follow_redirects=True,
            )
            response.raise_for_status()

            content = response.content

            self._validate_pdf_response(
                content=content,
                content_type=response.headers.get("content-type"),
            )

            local_file_path = self._write_pdf_file(
                pdf_asset_id=pdf_asset.pdf_asset_id,
                content=content,
            )

            return self._pdf_asset_repository.mark_downloaded(
                pdf_asset_id=pdf_asset.pdf_asset_id,
                local_file_path=str(local_file_path),
                sha256_checksum=self._calculate_sha256(content),
                content_type=response.headers.get(
                    "content-type",
                    "application/pdf",
                ),
                file_size_bytes=len(content),
            )

        except Exception as error:
            failure_message = f"{type(error).__name__}: {error}"

            self._pdf_asset_repository.mark_failed(
                pdf_asset_id=pdf_asset.pdf_asset_id,
                failure_message=failure_message,
            )

            raise PdfDownloadError(failure_message) from error

    def _write_pdf_file(
        self,
        *,
        pdf_asset_id: UUID,
        content: bytes,
    ) -> Path:
        """Write PDF bytes using a temporary file and atomic rename."""

        self._pdf_storage_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_path = self._pdf_storage_directory / f"{pdf_asset_id}.pdf"
        temporary_path = final_path.with_suffix(".pdf.tmp")

        temporary_path.write_bytes(content)
        temporary_path.replace(final_path)

        return final_path

    @staticmethod
    def _validate_pdf_response(
        *,
        content: bytes,
        content_type: str | None,
    ) -> None:
        """Validate that an HTTP response looks like a usable PDF."""

        if not content:
            raise PdfDownloadError("Downloaded file is empty.")

        if not content.startswith(b"%PDF"):
            raise PdfDownloadError("Downloaded file does not start with %PDF.")

        if content_type is None:
            return

        normalized_content_type = content_type.lower()

        if "application/pdf" not in normalized_content_type:
            raise PdfDownloadError(f"Unexpected content type: {content_type}")

    @staticmethod
    def _calculate_sha256(
        content: bytes,
    ) -> str:
        """Return SHA-256 checksum for downloaded bytes."""

        return hashlib.sha256(content).hexdigest()
