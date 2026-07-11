from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    PdfAssetRecord,
    TextChunkRecord,
)
from academic_literature_rag.models.text_chunk import TextChunk


class PdfAssetForChunkNotFoundError(LookupError):
    """Raised when text chunks reference a missing PDF asset."""


class TextChunkRepository:
    """Persists clean text chunks for PDF assets."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def replace_for_pdf_asset(
        self,
        *,
        pdf_asset_id: UUID,
        chunks: list[TextChunk],
    ) -> list[TextChunk]:
        """Replace all text chunks for one PDF asset."""

        with self._session_factory.begin() as session:
            self._ensure_pdf_asset_exists(
                session=session,
                pdf_asset_id=pdf_asset_id,
            )

            for chunk in chunks:
                if chunk.pdf_asset_id != pdf_asset_id:
                    raise ValueError("All text chunks must belong to the same PDF asset.")

            session.execute(
                delete(TextChunkRecord).where(TextChunkRecord.pdf_asset_id == str(pdf_asset_id))
            )

            for chunk in chunks:
                session.add(
                    TextChunkRecord(
                        id=str(chunk.text_chunk_id),
                        pdf_asset_id=str(chunk.pdf_asset_id),
                        chunk_index=chunk.chunk_index,
                        start_page_number=chunk.start_page_number,
                        end_page_number=chunk.end_page_number,
                        text=chunk.text,
                        char_count=chunk.char_count,
                        created_at=chunk.created_at,
                    )
                )

        return chunks

    def get(
        self,
        text_chunk_id: UUID,
    ) -> TextChunk | None:
        """Return one text chunk by ID."""

        with self._session_factory() as session:
            record = session.get(
                TextChunkRecord,
                str(text_chunk_id),
            )

        if record is None:
            return None

        return self._to_model(record)

    def list_for_pdf_asset(
        self,
        pdf_asset_id: UUID,
    ) -> list[TextChunk]:
        """Return text chunks for one PDF asset."""

        statement = (
            select(TextChunkRecord)
            .where(TextChunkRecord.pdf_asset_id == str(pdf_asset_id))
            .order_by(TextChunkRecord.chunk_index)
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
            raise PdfAssetForChunkNotFoundError(f"PDF asset does not exist: {pdf_asset_id}")

    @staticmethod
    def _to_model(
        record: TextChunkRecord,
    ) -> TextChunk:
        return TextChunk(
            text_chunk_id=UUID(record.id),
            pdf_asset_id=UUID(record.pdf_asset_id),
            chunk_index=record.chunk_index,
            start_page_number=record.start_page_number,
            end_page_number=record.end_page_number,
            text=record.text,
            char_count=record.char_count,
            created_at=TextChunkRepository._as_utc(record.created_at),
        )

    @staticmethod
    def _as_utc(
        value: datetime,
    ) -> datetime:
        """Return a timezone-aware UTC datetime from SQLite data."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)
