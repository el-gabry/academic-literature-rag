from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, delete, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    ChunkEmbeddingRecord,
    TextChunkRecord,
)
from academic_literature_rag.models.chunk_embedding import ChunkEmbedding
from academic_literature_rag.models.text_chunk import TextChunk


class TextChunkForEmbeddingNotFoundError(LookupError):
    """Raised when an embedding references a missing text chunk."""


class ChunkEmbeddingRepository:
    """Persists embedding vectors generated for text chunks."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        self._session_factory = session_factory

    def create_or_replace(
        self,
        embedding: ChunkEmbedding,
    ) -> ChunkEmbedding:
        """Create or replace an embedding for one text chunk and model."""

        with self._session_factory.begin() as session:
            self._ensure_text_chunk_exists(
                session=session,
                text_chunk_id=embedding.text_chunk_id,
            )

            session.execute(
                delete(ChunkEmbeddingRecord).where(
                    ChunkEmbeddingRecord.text_chunk_id == str(embedding.text_chunk_id),
                    ChunkEmbeddingRecord.embedding_model == embedding.embedding_model,
                )
            )

            session.add(
                ChunkEmbeddingRecord(
                    id=str(embedding.chunk_embedding_id),
                    text_chunk_id=str(embedding.text_chunk_id),
                    embedding_model=embedding.embedding_model,
                    embedding_vector_json=json.dumps(embedding.embedding_vector),
                    embedding_dimension=embedding.embedding_dimension,
                    created_at=embedding.created_at,
                )
            )

        return embedding

    def get_for_text_chunk(
        self,
        *,
        text_chunk_id: UUID,
        embedding_model: str,
    ) -> ChunkEmbedding | None:
        """Return one embedding for a text chunk and model."""

        normalized_model = self._normalize_embedding_model(embedding_model)

        statement = select(ChunkEmbeddingRecord).where(
            ChunkEmbeddingRecord.text_chunk_id == str(text_chunk_id),
            ChunkEmbeddingRecord.embedding_model == normalized_model,
        )

        with self._session_factory() as session:
            record = session.scalars(statement).one_or_none()

        if record is None:
            return None

        return self._to_embedding_model(record)

    def list_for_text_chunk(
        self,
        text_chunk_id: UUID,
    ) -> list[ChunkEmbedding]:
        """Return all embeddings stored for one text chunk."""

        statement = (
            select(ChunkEmbeddingRecord)
            .where(ChunkEmbeddingRecord.text_chunk_id == str(text_chunk_id))
            .order_by(ChunkEmbeddingRecord.embedding_model)
        )

        with self._session_factory() as session:
            records = session.scalars(statement).all()

        return [self._to_embedding_model(record) for record in records]

    def list_by_model(
        self,
        embedding_model: str,
    ) -> list[ChunkEmbedding]:
        """Return all embeddings stored for one embedding model."""

        normalized_model = self._normalize_embedding_model(embedding_model)

        statement = (
            select(ChunkEmbeddingRecord)
            .where(ChunkEmbeddingRecord.embedding_model == normalized_model)
            .order_by(ChunkEmbeddingRecord.created_at)
        )

        with self._session_factory() as session:
            records = session.scalars(statement).all()

        return [self._to_embedding_model(record) for record in records]

    def list_text_chunks_without_embedding(
        self,
        *,
        embedding_model: str,
        limit: int | None = None,
    ) -> list[TextChunk]:
        """Return text chunks that do not yet have an embedding for a model."""

        normalized_model = self._normalize_embedding_model(embedding_model)

        if limit is not None and limit < 1:
            raise ValueError("Missing embedding limit must be at least 1.")

        statement = (
            select(TextChunkRecord)
            .outerjoin(
                ChunkEmbeddingRecord,
                and_(
                    ChunkEmbeddingRecord.text_chunk_id == TextChunkRecord.id,
                    ChunkEmbeddingRecord.embedding_model == normalized_model,
                ),
            )
            .where(ChunkEmbeddingRecord.id.is_(None))
            .order_by(
                TextChunkRecord.created_at,
                TextChunkRecord.chunk_index,
            )
        )

        if limit is not None:
            statement = statement.limit(limit)

        with self._session_factory() as session:
            records = session.scalars(statement).all()

        return [self._to_text_chunk_model(record) for record in records]

    @staticmethod
    def _ensure_text_chunk_exists(
        *,
        session: Session,
        text_chunk_id: UUID,
    ) -> None:
        record = session.get(
            TextChunkRecord,
            str(text_chunk_id),
        )

        if record is None:
            raise TextChunkForEmbeddingNotFoundError(f"Text chunk does not exist: {text_chunk_id}")

    @staticmethod
    def _normalize_embedding_model(
        embedding_model: str,
    ) -> str:
        normalized_model = embedding_model.strip()

        if not normalized_model:
            raise ValueError("Embedding model cannot be empty.")

        return normalized_model

    @staticmethod
    def _to_embedding_model(
        record: ChunkEmbeddingRecord,
    ) -> ChunkEmbedding:
        return ChunkEmbedding(
            chunk_embedding_id=UUID(record.id),
            text_chunk_id=UUID(record.text_chunk_id),
            embedding_model=record.embedding_model,
            embedding_vector=[float(value) for value in json.loads(record.embedding_vector_json)],
            embedding_dimension=record.embedding_dimension,
            created_at=ChunkEmbeddingRepository._as_utc(record.created_at),
        )

    @staticmethod
    def _to_text_chunk_model(
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
            created_at=ChunkEmbeddingRepository._as_utc(record.created_at),
        )

    @staticmethod
    def _as_utc(
        value: datetime,
    ) -> datetime:
        """Return a timezone-aware UTC datetime from SQLite data."""

        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)

        return value.astimezone(UTC)
