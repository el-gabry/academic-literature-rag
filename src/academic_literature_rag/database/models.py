from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database tables."""


class SearchRunRecord(Base):
    """Database representation of one retrieval run."""

    __tablename__ = "search_runs"

    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(String(20), nullable=False)
    result_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    raw_response_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )


class CanonicalPaperRecord(Base):
    """Internal unified representation of one academic paper."""

    __tablename__ = "canonical_papers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(Text, nullable=False)

    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    authors_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    publication_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class SourcePaperRecord(Base):
    """A paper record as returned by one external source."""

    __tablename__ = "source_papers"

    __table_args__ = (
        UniqueConstraint(
            "source",
            "source_id",
            name="uq_source_papers_source_source_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    canonical_paper_id: Mapped[str | None] = mapped_column(
        ForeignKey(
            "canonical_papers.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    source: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[str] = mapped_column(String(255), nullable=False)

    title: Mapped[str] = mapped_column(Text, nullable=False)
    landing_url: Mapped[str] = mapped_column(Text, nullable=False)

    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    publication_year: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    open_access_pdf_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    citation_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )


class SearchRunPaperRecord(Base):
    """Links one retrieval run to a source paper found in that run."""

    __tablename__ = "search_run_papers"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("search_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    source_paper_id: Mapped[str] = mapped_column(
        ForeignKey("source_papers.id", ondelete="CASCADE"),
        primary_key=True,
    )

    result_position: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class PdfAssetRecord(Base):
    """Tracks one PDF candidate or downloaded PDF file."""

    __tablename__ = "pdf_assets"

    __table_args__ = (
        UniqueConstraint(
            "canonical_paper_id",
            "source_url",
            name="uq_pdf_assets_canonical_paper_source_url",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    canonical_paper_id: Mapped[str] = mapped_column(
        ForeignKey("canonical_papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("source_papers.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    source_url: Mapped[str] = mapped_column(Text, nullable=False)

    download_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    local_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    sha256_checksum: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    content_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    failure_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    downloaded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class PdfPageTextRecord(Base):
    """Stores extracted text for one page of a downloaded PDF."""

    __tablename__ = "pdf_page_texts"

    __table_args__ = (
        UniqueConstraint(
            "pdf_asset_id",
            "page_number",
            name="uq_pdf_page_texts_pdf_asset_page_number",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    pdf_asset_id: Mapped[str] = mapped_column(
        ForeignKey("pdf_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class TextChunkRecord(Base):
    """Stores clean text chunks ready for embedding and retrieval."""

    __tablename__ = "text_chunks"

    __table_args__ = (
        UniqueConstraint(
            "pdf_asset_id",
            "chunk_index",
            name="uq_text_chunks_pdf_asset_chunk_index",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    pdf_asset_id: Mapped[str] = mapped_column(
        ForeignKey("pdf_assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    start_page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    end_page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    char_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class ChunkEmbeddingRecord(Base):
    """Stores embedding vectors generated for text chunks."""

    __tablename__ = "chunk_embeddings"

    __table_args__ = (
        UniqueConstraint(
            "text_chunk_id",
            "embedding_model",
            name="uq_chunk_embeddings_text_chunk_model",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    text_chunk_id: Mapped[str] = mapped_column(
        ForeignKey("text_chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    embedding_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    embedding_vector_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
