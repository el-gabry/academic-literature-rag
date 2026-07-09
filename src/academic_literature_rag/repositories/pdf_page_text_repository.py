from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    PdfAssetRecord,
    PdfPageTextRecord,
)
from academic_literature_rag.models.pdf_page_text import PdfPageText


class PdfAssetForTextNotFoundError(LookupError):
    """Raised when extracted text references a missing PDF asset."""


class PdfPageTextRepository:
    """Persists extracted page text for downloaded PDF assets."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def replace_for_pdf_asset(
        self,
        *,
        pdf_asset_id: UUID,
        page_texts: list[PdfPageText],
    ) -> list[PdfPageText]:
        """Replace all extracted page text for one PDF asset."""

        with self._session_factory.begin() as session:
            self._ensure_pdf_asset_exists(
                session=session,
                pdf_asset_id=pdf_asset_id,
            )

            for page_text in page_texts:
                if page_text.pdf_asset_id != pdf_asset_id:
                    raise ValueError("All page-text records must belong to the same PDF asset.")

            session.execute(
                delete(PdfPageTextRecord).where(PdfPageTextRecord.pdf_asset_id == str(pdf_asset_id))
            )

            for page_text in page_texts:
                session.add(
                    PdfPageTextRecord(
                        id=str(page_text.pdf_page_text_id),
                        pdf_asset_id=str(page_text.pdf_asset_id),
                        page_number=page_text.page_number,
                        text=page_text.text,
                        created_at=page_text.created_at,
                    )
                )

        return page_texts

    def list_for_pdf_asset(
        self,
        pdf_asset_id: UUID,
    ) -> list[PdfPageText]:
        """Return extracted page text for one PDF asset."""

        statement = (
            select(PdfPageTextRecord)
            .where(PdfPageTextRecord.pdf_asset_id == str(pdf_asset_id))
            .order_by(PdfPageTextRecord.page_number)
        )

        with self._session_factory() as session:
            records = session.scalars(statement).all()

        return [self._to_model(record) for record in records]

    @staticmethod
    def _ensure_pdf_asset_exists(
        *,
        session: Session,
        pdf_asset_id: UUID,
    ) -> None:
        record = session.get(
            PdfAssetRecord,
            str(pdf_asset_id),
        )

        if record is None:
            raise PdfAssetForTextNotFoundError(f"PDF asset does not exist: {pdf_asset_id}")

    @staticmethod
    def _to_model(
        record: PdfPageTextRecord,
    ) -> PdfPageText:
        return PdfPageText(
            pdf_page_text_id=UUID(record.id),
            pdf_asset_id=UUID(record.pdf_asset_id),
            page_number=record.page_number,
            text=record.text,
            created_at=PdfPageTextRepository._as_utc(record.created_at),
        )

    @staticmethod
    def _as_utc(
        value: datetime,
    ) -> datetime:
        """Return a timezone-aware UTC datetime from SQLite data."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)
