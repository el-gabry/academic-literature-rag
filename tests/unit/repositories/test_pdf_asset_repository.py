from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
    PdfAssetRecord,
    SourcePaperRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    CanonicalPaperNotFoundError,
    PdfAssetRepository,
    SourcePaperForPdfNotFoundError,
)


def build_repository(
    database_path: Path,
) -> tuple[
    PdfAssetRepository,
    sessionmaker[Session],
]:
    """Create an isolated SQLite repository for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return PdfAssetRepository(session_factory), session_factory


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


def create_source_paper(
    *,
    session_factory: sessionmaker[Session],
    canonical_paper_id: UUID,
) -> UUID:
    """Insert one source-paper record directly for repository tests."""

    source_paper_id = uuid4()

    with session_factory.begin() as session:
        session.add(
            SourcePaperRecord(
                id=str(source_paper_id),
                canonical_paper_id=str(canonical_paper_id),
                source="arxiv",
                source_id="2302.01204v1",
                title="Example Paper",
                landing_url="https://arxiv.org/abs/2302.01204v1",
                abstract=None,
                authors_json=["A. Researcher"],
                publication_year=2026,
                venue=None,
                doi=None,
                arxiv_id="2302.01204",
                open_access_pdf_url="https://arxiv.org/pdf/2302.01204v1",
                citation_count=None,
            )
        )

    return source_paper_id


def test_create_or_get_pending_pdf_asset(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    source_paper_id = create_source_paper(
        session_factory=session_factory,
        canonical_paper_id=canonical_paper_id,
    )

    pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_paper_id=source_paper_id,
        source_url="https://arxiv.org/pdf/2302.01204v1",
    )

    assert pdf_asset.canonical_paper_id == canonical_paper_id
    assert pdf_asset.source_paper_id == source_paper_id
    assert pdf_asset.source_url == "https://arxiv.org/pdf/2302.01204v1"
    assert pdf_asset.download_status == "pending"
    assert pdf_asset.local_file_path is None

    assert repository.get(pdf_asset.pdf_asset_id) == pdf_asset


def test_create_or_get_pending_is_idempotent_for_same_paper_and_url(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    first_pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    second_pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    assert second_pdf_asset.pdf_asset_id == first_pdf_asset.pdf_asset_id

    with session_factory() as session:
        record_count = session.scalar(select(func.count()).select_from(PdfAssetRecord))

    assert record_count == 1


def test_existing_pdf_asset_can_be_updated_with_source_paper_id(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    first_pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    source_paper_id = create_source_paper(
        session_factory=session_factory,
        canonical_paper_id=canonical_paper_id,
    )

    second_pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_paper_id=source_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    assert second_pdf_asset.pdf_asset_id == first_pdf_asset.pdf_asset_id
    assert second_pdf_asset.source_paper_id == source_paper_id


def test_list_for_canonical_paper_returns_assets_in_creation_order(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    first_pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/first.pdf",
    )

    second_pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/second.pdf",
    )

    assert repository.list_for_canonical_paper(canonical_paper_id) == [
        first_pdf_asset,
        second_pdf_asset,
    ]


def test_mark_downloaded_updates_pdf_asset_metadata(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    downloaded_asset = repository.mark_downloaded(
        pdf_asset_id=pdf_asset.pdf_asset_id,
        local_file_path="data/pdfs/example.pdf",
        sha256_checksum="a" * 64,
        content_type="application/pdf",
        file_size_bytes=1024,
    )

    assert downloaded_asset.download_status == "downloaded"
    assert downloaded_asset.local_file_path == "data/pdfs/example.pdf"
    assert downloaded_asset.sha256_checksum == "a" * 64
    assert downloaded_asset.content_type == "application/pdf"
    assert downloaded_asset.file_size_bytes == 1024
    assert downloaded_asset.failure_message is None
    assert downloaded_asset.downloaded_at is not None


def test_mark_failed_updates_failure_message(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    failed_asset = repository.mark_failed(
        pdf_asset_id=pdf_asset.pdf_asset_id,
        failure_message="HTTP 404",
    )

    assert failed_asset.download_status == "failed"
    assert failed_asset.failure_message == "HTTP 404"
    assert failed_asset.downloaded_at is None


def test_missing_canonical_paper_raises_error(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    with pytest.raises(CanonicalPaperNotFoundError):
        repository.create_or_get_pending(
            canonical_paper_id=uuid4(),
            source_url="https://example.org/paper.pdf",
        )


def test_missing_source_paper_raises_error(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    canonical_paper_id = create_canonical_paper(session_factory)

    with pytest.raises(SourcePaperForPdfNotFoundError):
        repository.create_or_get_pending(
            canonical_paper_id=canonical_paper_id,
            source_paper_id=uuid4(),
            source_url="https://example.org/paper.pdf",
        )
