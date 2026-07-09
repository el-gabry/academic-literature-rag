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
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.repositories.pdf_page_text_repository import (
    PdfPageTextRepository,
)
from academic_literature_rag.services import pdf_text_extraction_service
from academic_literature_rag.services.pdf_text_extraction_service import (
    PdfTextExtractionError,
    PdfTextExtractionService,
)


class FakePdfPage:
    """Fake PDF page used to avoid depending on real PDF text rendering."""

    def __init__(
        self,
        text: str,
    ) -> None:
        self._text = text

    def extract_text(
        self,
    ) -> str:
        return self._text


class FakePdfReader:
    """Fake pypdf reader with two readable pages."""

    def __init__(
        self,
        pdf_path: Path,
    ) -> None:
        assert pdf_path.exists()

        self.pages = [
            FakePdfPage(" First   page text\n\nwith   spacing "),
            FakePdfPage("Second page\nwith more text"),
        ]


class EmptyFakePdfReader:
    """Fake pypdf reader that returns no readable text."""

    def __init__(
        self,
        pdf_path: Path,
    ) -> None:
        assert pdf_path.exists()

        self.pages = [
            FakePdfPage("   "),
            FakePdfPage(""),
        ]


def build_repositories(
    database_path: Path,
) -> tuple[
    PdfAssetRepository,
    PdfPageTextRepository,
    sessionmaker[Session],
]:
    """Create isolated repositories for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return (
        PdfAssetRepository(session_factory),
        PdfPageTextRepository(session_factory),
        session_factory,
    )


def create_canonical_paper(
    session_factory: sessionmaker[Session],
) -> UUID:
    """Insert one canonical-paper record directly for service tests."""

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


def create_downloaded_pdf_asset(
    *,
    pdf_asset_repository: PdfAssetRepository,
    session_factory: sessionmaker[Session],
    pdf_path: Path,
) -> UUID:
    """Create one downloaded PDF asset and return its ID."""

    canonical_paper_id = create_canonical_paper(session_factory)

    pdf_asset = pdf_asset_repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    downloaded_asset = pdf_asset_repository.mark_downloaded(
        pdf_asset_id=pdf_asset.pdf_asset_id,
        local_file_path=str(pdf_path),
        sha256_checksum="a" * 64,
        content_type="application/pdf",
        file_size_bytes=pdf_path.stat().st_size,
    )

    return downloaded_asset.pdf_asset_id


def create_pending_pdf_asset(
    *,
    pdf_asset_repository: PdfAssetRepository,
    session_factory: sessionmaker[Session],
) -> UUID:
    """Create one pending PDF asset and return its ID."""

    canonical_paper_id = create_canonical_paper(session_factory)

    pdf_asset = pdf_asset_repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url="https://example.org/paper.pdf",
    )

    return pdf_asset.pdf_asset_id


def build_service(
    *,
    pdf_asset_repository: PdfAssetRepository,
    pdf_page_text_repository: PdfPageTextRepository,
) -> PdfTextExtractionService:
    """Create the PDF text extraction service."""

    return PdfTextExtractionService(
        pdf_asset_repository=pdf_asset_repository,
        pdf_page_text_repository=pdf_page_text_repository,
    )


def test_extract_persists_page_level_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_asset_repository, pdf_page_text_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pdf_asset_id = create_downloaded_pdf_asset(
        pdf_asset_repository=pdf_asset_repository,
        session_factory=session_factory,
        pdf_path=pdf_path,
    )

    monkeypatch.setattr(
        pdf_text_extraction_service,
        "PdfReader",
        FakePdfReader,
    )

    service = build_service(
        pdf_asset_repository=pdf_asset_repository,
        pdf_page_text_repository=pdf_page_text_repository,
    )

    page_texts = service.extract(pdf_asset_id)

    assert len(page_texts) == 2

    assert page_texts[0].pdf_asset_id == pdf_asset_id
    assert page_texts[0].page_number == 1
    assert page_texts[0].text == "First page text\nwith spacing"

    assert page_texts[1].pdf_asset_id == pdf_asset_id
    assert page_texts[1].page_number == 2
    assert page_texts[1].text == "Second page\nwith more text"

    assert pdf_page_text_repository.list_for_pdf_asset(pdf_asset_id) == page_texts


def test_extract_replaces_existing_page_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_asset_repository, pdf_page_text_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pdf_asset_id = create_downloaded_pdf_asset(
        pdf_asset_repository=pdf_asset_repository,
        session_factory=session_factory,
        pdf_path=pdf_path,
    )

    monkeypatch.setattr(
        pdf_text_extraction_service,
        "PdfReader",
        FakePdfReader,
    )

    service = build_service(
        pdf_asset_repository=pdf_asset_repository,
        pdf_page_text_repository=pdf_page_text_repository,
    )

    first_extraction = service.extract(pdf_asset_id)
    second_extraction = service.extract(pdf_asset_id)

    assert len(first_extraction) == 2
    assert len(second_extraction) == 2
    assert pdf_page_text_repository.list_for_pdf_asset(pdf_asset_id) == second_extraction


def test_extract_rejects_non_downloaded_pdf_asset(
    tmp_path: Path,
) -> None:
    pdf_asset_repository, pdf_page_text_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_asset_id = create_pending_pdf_asset(
        pdf_asset_repository=pdf_asset_repository,
        session_factory=session_factory,
    )

    service = build_service(
        pdf_asset_repository=pdf_asset_repository,
        pdf_page_text_repository=pdf_page_text_repository,
    )

    with pytest.raises(
        PdfTextExtractionError,
        match="must be downloaded",
    ):
        service.extract(pdf_asset_id)


def test_extract_rejects_missing_pdf_file(
    tmp_path: Path,
) -> None:
    pdf_asset_repository, pdf_page_text_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    temporary_pdf_path = tmp_path / "temporary.pdf"
    temporary_pdf_path.write_bytes(b"%PDF-1.4\n")

    temporary_pdf_path = tmp_path / "temporary.pdf"
    temporary_pdf_path.write_bytes(b"%PDF-1.4\n")

    missing_pdf_path = tmp_path / "missing.pdf"

    pdf_asset_id = create_downloaded_pdf_asset(
        pdf_asset_repository=pdf_asset_repository,
        session_factory=session_factory,
        pdf_path=temporary_pdf_path,
    )

    with session_factory.begin() as session:
        record = session.get(
            PdfAssetRecord,
            str(pdf_asset_id),
        )

        assert record is not None

        record.local_file_path = str(missing_pdf_path)

    service = build_service(
        pdf_asset_repository=pdf_asset_repository,
        pdf_page_text_repository=pdf_page_text_repository,
    )

    with pytest.raises(
        PdfTextExtractionError,
        match="does not exist",
    ):
        service.extract(pdf_asset_id)


def test_extract_rejects_pdf_with_no_readable_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_asset_repository, pdf_page_text_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db"
    )

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    pdf_asset_id = create_downloaded_pdf_asset(
        pdf_asset_repository=pdf_asset_repository,
        session_factory=session_factory,
        pdf_path=pdf_path,
    )

    monkeypatch.setattr(
        pdf_text_extraction_service,
        "PdfReader",
        EmptyFakePdfReader,
    )

    service = build_service(
        pdf_asset_repository=pdf_asset_repository,
        pdf_page_text_repository=pdf_page_text_repository,
    )

    with pytest.raises(
        PdfTextExtractionError,
        match="no readable text",
    ):
        service.extract(pdf_asset_id)
