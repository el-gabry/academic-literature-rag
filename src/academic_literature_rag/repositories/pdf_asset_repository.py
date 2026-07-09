from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    PdfAssetRecord,
    SourcePaperRecord,
)
from academic_literature_rag.models.pdf_asset import PdfAsset


class CanonicalPaperNotFoundError(LookupError):
    """Raised when a PDF asset references a missing canonical paper."""


class SourcePaperForPdfNotFoundError(LookupError):
    """Raised when a PDF asset references a missing source paper."""


class PdfAssetNotFoundError(LookupError):
    """Raised when a requested PDF asset does not exist."""


class PdfAssetRepository:
    """Persists PDF asset records before and after downloading."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def create_or_get_pending(
        self,
        *,
        canonical_paper_id: UUID,
        source_url: str,
        source_paper_id: UUID | None = None,
    ) -> PdfAsset:
        """Create or return a pending PDF asset for one canonical paper."""

        cleaned_source_url = source_url.strip()

        with self._session_factory.begin() as session:
            self._ensure_canonical_paper_exists(
                session=session,
                canonical_paper_id=canonical_paper_id,
            )

            if source_paper_id is not None:
                self._ensure_source_paper_exists(
                    session=session,
                    source_paper_id=source_paper_id,
                )

            existing_record = session.scalar(
                select(PdfAssetRecord).where(
                    PdfAssetRecord.canonical_paper_id == str(canonical_paper_id),
                    PdfAssetRecord.source_url == cleaned_source_url,
                )
            )

            if existing_record is not None:
                if existing_record.source_paper_id is None and source_paper_id is not None:
                    existing_record.source_paper_id = str(source_paper_id)

                return self._to_model(existing_record)

            pdf_asset = PdfAsset(
                canonical_paper_id=canonical_paper_id,
                source_paper_id=source_paper_id,
                source_url=cleaned_source_url,
            )

            record = PdfAssetRecord(
                id=str(pdf_asset.pdf_asset_id),
                canonical_paper_id=str(pdf_asset.canonical_paper_id),
                source_paper_id=(
                    str(pdf_asset.source_paper_id)
                    if pdf_asset.source_paper_id is not None
                    else None
                ),
                source_url=pdf_asset.source_url,
                download_status=pdf_asset.download_status,
                local_file_path=pdf_asset.local_file_path,
                sha256_checksum=pdf_asset.sha256_checksum,
                content_type=pdf_asset.content_type,
                file_size_bytes=pdf_asset.file_size_bytes,
                failure_message=pdf_asset.failure_message,
                created_at=pdf_asset.created_at,
                downloaded_at=pdf_asset.downloaded_at,
            )

            session.add(record)

            return pdf_asset

    def mark_downloaded(
        self,
        *,
        pdf_asset_id: UUID,
        local_file_path: str,
        sha256_checksum: str,
        content_type: str,
        file_size_bytes: int,
    ) -> PdfAsset:
        """Mark a PDF asset as successfully downloaded."""

        with self._session_factory.begin() as session:
            record = self._get_existing_record(
                session=session,
                pdf_asset_id=pdf_asset_id,
            )

            record.download_status = "downloaded"
            record.local_file_path = local_file_path
            record.sha256_checksum = sha256_checksum
            record.content_type = content_type
            record.file_size_bytes = file_size_bytes
            record.failure_message = None
            record.downloaded_at = datetime.now(UTC)

            return self._to_model(record)

    def mark_failed(
        self,
        *,
        pdf_asset_id: UUID,
        failure_message: str,
    ) -> PdfAsset:
        """Mark a PDF asset as failed."""

        with self._session_factory.begin() as session:
            record = self._get_existing_record(
                session=session,
                pdf_asset_id=pdf_asset_id,
            )

            record.download_status = "failed"
            record.failure_message = failure_message
            record.downloaded_at = None

            return self._to_model(record)

    def get(
        self,
        pdf_asset_id: UUID,
    ) -> PdfAsset | None:
        """Return one PDF asset by ID."""

        with self._session_factory() as session:
            record = session.get(
                PdfAssetRecord,
                str(pdf_asset_id),
            )

        if record is None:
            return None

        return self._to_model(record)

    def list_pending(
        self,
        *,
        limit: int | None = None,
    ) -> list[PdfAsset]:
        """Return pending PDF assets ordered by creation time."""

        if limit is not None and limit < 1:
            raise ValueError("Pending PDF asset limit must be at least 1.")
        statement = (
            select(PdfAssetRecord)
            .where(PdfAssetRecord.download_status == "pending")
            .order_by(PdfAssetRecord.created_at)
        )
        if limit is not None:
            statement = statement.limit(limit)

        with self._session_factory() as session:
            records = session.scalars(statement).all()

        return [self._to_model(record) for record in records]

    def list_for_canonical_paper(
        self,
        canonical_paper_id: UUID,
    ) -> list[PdfAsset]:
        """Return all PDF assets attached to one canonical paper."""

        statement = (
            select(PdfAssetRecord)
            .where(PdfAssetRecord.canonical_paper_id == str(canonical_paper_id))
            .order_by(PdfAssetRecord.created_at)
        )

        with self._session_factory() as session:
            records = session.scalars(statement).all()

        return [self._to_model(record) for record in records]

    @staticmethod
    def _ensure_canonical_paper_exists(
        *,
        session: Session,
        canonical_paper_id: UUID,
    ) -> None:
        record = session.get(
            CanonicalPaperRecord,
            str(canonical_paper_id),
        )

        if record is None:
            raise CanonicalPaperNotFoundError(
                f"Canonical paper does not exist: {canonical_paper_id}"
            )

    @staticmethod
    def _ensure_source_paper_exists(
        *,
        session: Session,
        source_paper_id: UUID,
    ) -> None:
        record = session.get(
            SourcePaperRecord,
            str(source_paper_id),
        )

        if record is None:
            raise SourcePaperForPdfNotFoundError(f"Source paper does not exist: {source_paper_id}")

    @staticmethod
    def _get_existing_record(
        *,
        session: Session,
        pdf_asset_id: UUID,
    ) -> PdfAssetRecord:
        record = session.get(
            PdfAssetRecord,
            str(pdf_asset_id),
        )

        if record is None:
            raise PdfAssetNotFoundError(f"PDF asset does not exist: {pdf_asset_id}")

        return record

    @staticmethod
    def _to_model(
        record: PdfAssetRecord,
    ) -> PdfAsset:
        return PdfAsset(
            pdf_asset_id=UUID(record.id),
            canonical_paper_id=UUID(record.canonical_paper_id),
            source_paper_id=(
                UUID(record.source_paper_id) if record.source_paper_id is not None else None
            ),
            source_url=record.source_url,
            download_status=record.download_status,
            local_file_path=record.local_file_path,
            sha256_checksum=record.sha256_checksum,
            content_type=record.content_type,
            file_size_bytes=record.file_size_bytes,
            failure_message=record.failure_message,
            created_at=PdfAssetRepository._as_utc(record.created_at),
            downloaded_at=(
                PdfAssetRepository._as_utc(record.downloaded_at)
                if record.downloaded_at is not None
                else None
            ),
        )

    @staticmethod
    def _as_utc(
        value: datetime,
    ) -> datetime:
        """Return a timezone-aware UTC datetime from SQLite data."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)
