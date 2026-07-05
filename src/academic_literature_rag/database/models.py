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

    raw_response_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


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
