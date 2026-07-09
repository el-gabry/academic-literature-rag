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
from academic_literature_rag.models.pdf_page_text import PdfPageText
from academic_literature_rag.repositories.pdf_page_text_repository import (
    PdfAssetForTextNotFoundError,
    PdfPageTextRepository,
)


def build_repository(
    database_path: Path,
) -> tuple[
    PdfPageTextRepository,
    sessionmaker[Session],
]:
    """Create an isolated PDF page-text repository for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return PdfPageTextRepository(session_factory), session_factory


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


def test_replace_for_pdf_asset_saves_page_texts(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    page_texts = [
        PdfPageText(
            pdf_asset_id=pdf_asset_id,
            page_number=1,
            text="First page text.",
        ),
        PdfPageText(
            pdf_asset_id=pdf_asset_id,
            page_number=2,
            text="Second page text.",
        ),
    ]

    saved_page_texts = repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        page_texts=page_texts,
    )

    assert saved_page_texts == page_texts
    assert repository.list_for_pdf_asset(pdf_asset_id) == page_texts


def test_replace_for_pdf_asset_replaces_existing_texts(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        page_texts=[
            PdfPageText(
                pdf_asset_id=pdf_asset_id,
                page_number=1,
                text="Old text.",
            )
        ],
    )

    new_page_texts = [
        PdfPageText(
            pdf_asset_id=pdf_asset_id,
            page_number=1,
            text="New first page text.",
        ),
        PdfPageText(
            pdf_asset_id=pdf_asset_id,
            page_number=2,
            text="New second page text.",
        ),
    ]

    repository.replace_for_pdf_asset(
        pdf_asset_id=pdf_asset_id,
        page_texts=new_page_texts,
    )

    assert repository.list_for_pdf_asset(pdf_asset_id) == new_page_texts


def test_list_for_pdf_asset_returns_empty_list_when_no_text_exists(
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

    with pytest.raises(PdfAssetForTextNotFoundError):
        repository.replace_for_pdf_asset(
            pdf_asset_id=missing_pdf_asset_id,
            page_texts=[],
        )


def test_replace_rejects_page_texts_for_different_pdf_asset(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pdf_asset(session_factory)

    with pytest.raises(ValueError):
        repository.replace_for_pdf_asset(
            pdf_asset_id=pdf_asset_id,
            page_texts=[
                PdfPageText(
                    pdf_asset_id=uuid4(),
                    page_number=1,
                    text="Wrong PDF asset.",
                )
            ],
        )
