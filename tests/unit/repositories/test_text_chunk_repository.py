from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    PdfAssetRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.text_chunk import TextChunk
from academic_literature_rag.repositories.text_chunk_repository import (
    PdfAssetForChunkNotFoundError,
    TextChunkRepository,
)


def build_repository(
    database_path: Path,
) -> tuple[
    TextChunkRepository,
    sessionmaker[Session],
]:
    """Create an isolated text chunk repository for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return TextChunkRepository(session_factory), session_factory


def create_canonical_paper(
    session_factory: sessionmaker[Session],
) -> UUID:
    """Insert one canonical-paper record directly for repository tests."""

    canonical_paper_id = uuid4()

    with session_factory.begin() as session:
        session.add(
            CanonicalPaperRecord(
                id=str(canonical_paper_id),
                title="Example Paper",
                normalized_title="example paper",
                doi="10.1000/example",
                arxiv_id=None,
                authors_json=["A. Researcher"],
                publication_year=2026,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

    return canonical_paper_id


def create_pdf_asset(
    session_factory: sessionmaker[Session],
) -> UUID:
    """Insert one downloaded PDF asset directly for repository tests."""

    canonical_paper_id = create_canonical_paper(session_factory)
    pdf_asset_id = uuid4()

    with session_factory.begin() as session:
        session.add(
            PdfAssetRecord(
                id=str(pdf_asset_id),
                canonical_paper_id=str(canonical_paper_id),
                source_paper_id=None,
                source_url="https://example.org/paper.pdf",
                download_status="downloaded",
                local_file_path="data/pdfs/example.pdf",
                sha256_checksum="a" * 64,
                content_type="application/pdf",
                file_size_bytes=1024,
                failure_message=None,
                created_at=datetime.now(UTC),
                downloaded_at=datetime.now(UTC),
            )
        )

    return pdf_asset_id


def test_replace_for_pdf_asset_saves_chunks(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    chunks = [
        TextChunk(
            pdf_asset_id=pdf_asset_id,
            chunk_index=0,
            start_page_number=1,
            end_page_number=1,
            text="First chunk text.",
            char_count=len("First chunk text."),
        ),
        TextChunk(
            pdf_asset_id=pdf_asset_id,
            chunk_index=1,
            start_page_number=1,
            end_page_number=2,
            text="Second chunk text.",
            char_count=len("Second chunk text."),
        ),
    ]

    saved_chunks = repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        chunks=chunks,
    )

    assert saved_chunks == chunks
    assert repository.list_for_pdf_asset(pdf_asset_id) == chunks


def test_replace_for_pdf_asset_replaces_existing_chunks(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        chunks=[
            TextChunk(
                pdf_asset_id=pdf_asset_id,
                chunk_index=0,
                start_page_number=1,
                end_page_number=1,
                text="Old chunk text.",
                char_count=len("Old chunk text."),
            )
        ],
    )

    new_chunks = [
        TextChunk(
            pdf_asset_id=pdf_asset_id,
            chunk_index=0,
            start_page_number=2,
            end_page_number=2,
            text="New chunk text.",
            char_count=len("New chunk text."),
        )
    ]

    repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        chunks=new_chunks,
    )

    assert repository.list_for_pdf_asset(pdf_asset_id) == new_chunks


def test_list_for_pdf_asset_returns_empty_list_when_no_chunks_exist(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    assert repository.list_for_pdf_asset(pdf_asset_id) == []


def test_replace_for_missing_pdf_asset_raises_error(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    missing_pdf_asset_id = uuid4()

    with pytest.raises(PdfAssetForChunkNotFoundError):
        repository.replace_for_pdf_asset(
            pdf_asset_id=missing_pdf_asset_id,
            chunks=[],
        )


def test_replace_rejects_chunks_for_different_pdf_asset(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    with pytest.raises(ValueError):
        repository.replace_for_pdf_asset(
            pdf_asset_id=pdf_asset_id,
            chunks=[
                TextChunk(
                    pdf_asset_id=uuid4(),
                    chunk_index=0,
                    start_page_number=1,
                    end_page_number=1,
                    text="Wrong PDF asset chunk.",
                    char_count=len("Wrong PDF asset chunk."),
                )
            ],
        )
