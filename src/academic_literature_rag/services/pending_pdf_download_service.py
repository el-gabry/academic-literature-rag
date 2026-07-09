from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from academic_literature_rag.models.pdf_asset import PdfAsset
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.services.pdf_download_service import (
    PdfDownloadError,
    PdfDownloadService,
)


@dataclass(frozen=True)
class PendingPdfDownloadResult:
    """Summary of one pending PDF download attempt."""

    pdf_asset_id: UUID
    source_url: str
    status: str
    error_message: str | None = None


class PendingPdfDownloadService:
    """Downloads pending PDF assets without stopping on individual failures."""

    def __init__(
        self,
        *,
        pdf_asset_repository: PdfAssetRepository,
        pdf_download_service: PdfDownloadService,
    ) -> None:
        self._pdf_asset_repository = pdf_asset_repository
        self._pdf_download_service = pdf_download_service

    def download_pending(
        self,
        *,
        limit: int | None = None,
    ) -> list[PendingPdfDownloadResult]:
        """Download pending PDF assets and return one result per asset."""

        pending_assets = self._pdf_asset_repository.list_pending(
            limit=limit,
        )

        return [self._download_one(pdf_asset) for pdf_asset in pending_assets]

    def _download_one(
        self,
        pdf_asset: PdfAsset,
    ) -> PendingPdfDownloadResult:
        """Download one PDF asset and convert success/failure into a result."""

        try:
            downloaded_asset = self._pdf_download_service.download(pdf_asset.pdf_asset_id)

            return PendingPdfDownloadResult(
                pdf_asset_id=downloaded_asset.pdf_asset_id,
                source_url=downloaded_asset.source_url,
                status=downloaded_asset.download_status,
            )

        except PdfDownloadError as error:
            failed_asset = self._pdf_asset_repository.get(pdf_asset.pdf_asset_id)

            return PendingPdfDownloadResult(
                pdf_asset_id=pdf_asset.pdf_asset_id,
                source_url=pdf_asset.source_url,
                status=(failed_asset.download_status if failed_asset is not None else "failed"),
                error_message=str(error),
            )
