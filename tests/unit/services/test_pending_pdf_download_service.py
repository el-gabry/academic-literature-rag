from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import CanonicalPaperRecord
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.services.pdf_download_service import (
    PdfDownloadService,
)
from academic_literature_rag.services.pending_pdf_download_service import (
    PendingPdfDownloadService,
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
    source_url: str,
) -> UUID:
    """Create one pending PDF asset and return its ID."""

    canonical_paper_id = create_canonical_paper(session_factory)

    pdf_asset = repository.create_or_get_pending(
        canonical_paper_id=canonical_paper_id,
        source_url=source_url,
    )

    return pdf_asset.pdf_asset_id


def build_pending_download_service(
    *,
    repository: PdfAssetRepository,
    pdf_storage_directory: Path,
    http_client: httpx.Client,
) -> PendingPdfDownloadService:
    """Create the pending-download workflow service."""

    pdf_download_service = PdfDownloadService(
        pdf_asset_repository=repository,
        pdf_storage_directory=pdf_storage_directory,
        http_client=http_client,
    )

    return PendingPdfDownloadService(
        pdf_asset_repository=repository,
        pdf_download_service=pdf_download_service,
    )


def test_download_pending_downloads_valid_assets_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    valid_pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
        source_url="https://example.org/valid.pdf",
    )

    invalid_pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
        source_url="https://example.org/invalid.pdf",
    )

    valid_pdf_content = b"%PDF-1.4\n% valid pdf bytes\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://example.org/valid.pdf":
            return httpx.Response(
                status_code=200,
                headers={"content-type": "application/pdf"},
                content=valid_pdf_content,
                request=request,
            )

        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/html"},
            content=b"%PDF-1.4\n",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = build_pending_download_service(
            repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        results = service.download_pending()

    results_by_id = {result.pdf_asset_id: result for result in results}

    assert set(results_by_id) == {
        valid_pdf_asset_id,
        invalid_pdf_asset_id,
    }

    assert results_by_id[valid_pdf_asset_id].status == "downloaded"
    assert results_by_id[valid_pdf_asset_id].error_message is None

    assert results_by_id[invalid_pdf_asset_id].status == "failed"
    assert results_by_id[invalid_pdf_asset_id].error_message is not None

    valid_asset = repository.get(valid_pdf_asset_id)
    invalid_asset = repository.get(invalid_pdf_asset_id)

    assert valid_asset is not None
    assert valid_asset.download_status == "downloaded"
    assert valid_asset.local_file_path is not None
    assert Path(valid_asset.local_file_path).exists()

    assert invalid_asset is not None
    assert invalid_asset.download_status == "failed"
    assert invalid_asset.failure_message is not None
    assert "Unexpected content type" in invalid_asset.failure_message


def test_download_pending_respects_limit(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    first_pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
        source_url="https://example.org/first.pdf",
    )

    second_pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
        source_url="https://example.org/second.pdf",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.4\n",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = build_pending_download_service(
            repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        results = service.download_pending(limit=1)

    assert len(results) == 1

    first_asset = repository.get(first_pdf_asset_id)
    second_asset = repository.get(second_pdf_asset_id)

    assert first_asset is not None
    assert second_asset is not None

    downloaded_count = sum(
        asset.download_status == "downloaded" for asset in [first_asset, second_asset]
    )

    pending_count = sum(asset.download_status == "pending" for asset in [first_asset, second_asset])

    assert downloaded_count == 1
    assert pending_count == 1


def test_download_pending_ignores_already_downloaded_assets(
    tmp_path: Path,
) -> None:
    repository, session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    downloaded_pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
        source_url="https://example.org/already-downloaded.pdf",
    )

    pending_pdf_asset_id = create_pending_pdf_asset(
        repository=repository,
        session_factory=session_factory,
        source_url="https://example.org/pending.pdf",
    )

    repository.mark_downloaded(
        pdf_asset_id=downloaded_pdf_asset_id,
        local_file_path="data/pdfs/already-downloaded.pdf",
        sha256_checksum="a" * 64,
        content_type="application/pdf",
        file_size_bytes=100,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.org/pending.pdf"

        return httpx.Response(
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.4\n",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        service = build_pending_download_service(
            repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        results = service.download_pending()

    assert len(results) == 1
    assert results[0].pdf_asset_id == pending_pdf_asset_id
    assert results[0].status == "downloaded"

    downloaded_asset = repository.get(downloaded_pdf_asset_id)
    pending_asset = repository.get(pending_pdf_asset_id)

    assert downloaded_asset is not None
    assert pending_asset is not None

    assert downloaded_asset.download_status == "downloaded"
    assert pending_asset.download_status == "downloaded"


def test_download_pending_with_no_pending_assets_returns_empty_list(
    tmp_path: Path,
) -> None:
    repository, _session_factory = build_repository(tmp_path / "academic_literature_rag.db")

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                status_code=500,
                request=request,
            )
        )
    ) as http_client:
        service = build_pending_download_service(
            repository=repository,
            pdf_storage_directory=tmp_path / "pdfs",
            http_client=http_client,
        )

        results = service.download_pending()

    assert results == []
