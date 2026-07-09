from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import CanonicalPaperRecord
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetNotFoundError,
    PdfAssetRepository,
)
from academic_literature_rag.services.pdf_download_service import (
    PdfDownloadError,
    PdfDownloadService,
)


def build_repository(
    database_path: Path,
) -> tuple[
    PdfAssetRepository,
    sessionmaker[Session],
]:
    """Create an isolated PDF asset repository for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return PdfAssetRepository(session_factory), session_factory


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


def create_pending_pdf_asset(
    *,
    repository: PdfAssetRepository,
    session_factory: sessionmaker[Session],
    source_url: str = "https://example.org/paper.pdf",
) -> UUID:
    """Create one pending PDF asset and return its ID."""

    canonical_paper_id = create_canonical_paper(session_factory)

    pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url=source_url,
    )

    return pdf_asset.pdf_asset_id


def test_download_valid_pdf_marks_asset_downloaded(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
    )

    pdf_content = b"%PDF-1.4\n% example pdf bytes\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.org/paper.pdf"

        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=pdf_content,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = PdfDownloadService(
            pdf_asset_repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        downloaded_asset = service.download(pdf_asset_id)

    assert downloaded_asset.download_status == "downloaded"
    assert downloaded_asset.local_file_path is not None
    assert downloaded_asset.local_file_path.endswith(f"{pdf_asset_id}.pdf")
    assert downloaded_asset.content_type == "application/pdf"
    assert downloaded_asset.file_size_bytes == len(pdf_content)
    assert downloaded_asset.sha256_checksum is not None
    assert len(downloaded_asset.sha256_checksum) == 64
    assert downloaded_asset.failure_message is None
    assert downloaded_asset.downloaded_at is not None

    saved_file = Path(downloaded_asset.local_file_path)

    assert saved_file.exists()
    assert saved_file.read_bytes() == pdf_content


def test_download_rejects_non_pdf_bytes_and_marks_asset_failed(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"not a pdf",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = PdfDownloadService(
            pdf_asset_repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        with pytest.raises(PdfDownloadError):
            service.download(pdf_asset_id)

    failed_asset = repository.get(pdf_asset_id)

    assert failed_asset is not None
    assert failed_asset.download_status == "failed"
    assert failed_asset.failure_message is not None
    assert "does not start with %PDF" in failed_asset.failure_message


def test_download_rejects_unexpected_content_type_and_marks_asset_failed(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/html"},
            content=b"%PDF-1.4\n",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = PdfDownloadService(
            pdf_asset_repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        with pytest.raises(PdfDownloadError):
            service.download(pdf_asset_id)

    failed_asset = repository.get(pdf_asset_id)

    assert failed_asset is not None
    assert failed_asset.download_status == "failed"
    assert failed_asset.failure_message is not None
    assert "Unexpected content type" in failed_asset.failure_message


def test_download_http_error_marks_asset_failed(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            text="not found",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = PdfDownloadService(
            pdf_asset_repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        with pytest.raises(PdfDownloadError):
            service.download(pdf_asset_id)

    failed_asset = repository.get(pdf_asset_id)

    assert failed_asset is not None
    assert failed_asset.download_status == "failed"
    assert failed_asset.failure_message is not None
    assert "HTTPStatusError" in failed_asset.failure_message


def test_download_missing_asset_raises_not_found_error(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    service = PdfDownloadService(
        pdf_asset_repository=repository,
        pdf_storage_directory=tmp_path / "pdfs",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    status_code=200,
                    request=request,
                )
            )
        ),
    )

    with pytest.raises(PdfAssetNotFoundError):
        service.download(uuid4())

    service.close()
