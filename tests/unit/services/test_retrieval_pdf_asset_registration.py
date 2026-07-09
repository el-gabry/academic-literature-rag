from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    PdfAssetRecord,
    SourcePaperRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.repositories.canonical_paper_repository import (
    CanonicalPaperRepository,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)
from academic_literature_rag.repositories.source_paper_repository import (
    SourcePaperRepository,
)
from academic_literature_rag.services.persisted_retrieval_service import (
    PersistedRetrievalService,
)
from academic_literature_rag.storage.raw_response_store import RawResponseStore


class StaticPaperSourceClient:
    """Deterministic connector used for retrieval workflow tests."""

    raw_extension = "json"

    def __init__(
        self,
        *,
        source_name: str,
        papers: list[PaperCandidate],
    ) -> None:
        self.source_name = source_name
        self._papers = papers

    def fetch_raw_response(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> str:
        assert query
        assert limit > 0

        return "{}"

    def map_raw_response(
        self,
        raw_response: str,
    ) -> list[PaperCandidate]:
        assert raw_response == "{}"

        return list(self._papers)


def build_service(
    *,
    database_path: Path,
    raw_directory: Path,
    client: StaticPaperSourceClient,
) -> tuple[
    PersistedRetrievalService,
    sessionmaker[Session],
]:
    """Create one isolated persisted retrieval service."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    service = PersistedRetrievalService(
        client=client,
        raw_response_store=RawResponseStore(raw_directory),
        search_run_repository=SearchRunRepository(session_factory),
        source_paper_repository=SourcePaperRepository(session_factory),
        canonical_paper_repository=CanonicalPaperRepository(session_factory),
        pdf_asset_repository=PdfAssetRepository(session_factory),
    )

    return service, session_factory


def make_candidate(
    *,
    source: str,
    source_id: str,
    title: str,
    open_access_pdf_url: str | None,
) -> PaperCandidate:
    """Build one source paper candidate."""

    return PaperCandidate(
        source=source,
        source_id=source_id,
        title=title,
        landing_url=f"https://example.org/{source}/{source_id}",
        retrieved_at=datetime.now(UTC),
        authors=["A. Researcher"],
        publication_year=2026,
        doi="10.1000/example-paper",
        open_access_pdf_url=open_access_pdf_url,
    )


def test_retrieval_registers_open_access_pdf_url_as_pending_asset(
    tmp_path: Path,
) -> None:
    client = StaticPaperSourceClient(
        source_name="arxiv",
        papers=[
            make_candidate(
                source="arxiv",
                source_id="2302.01204v1",
                title="Example Paper With PDF",
                open_access_pdf_url="https://arxiv.org/pdf/2302.01204v1",
            )
        ],
    )

    service, session_factory = build_service(
        database_path=tmp_path / "academic_literature_rag.db",
        raw_directory=tmp_path / "raw",
        client=client,
    )

    result = service.search(
        query="example paper",
        limit=1,
    )

    assert result.run.status == "completed"

    with session_factory() as session:
        source_paper = session.scalar(
            select(SourcePaperRecord).where(
                SourcePaperRecord.source == "arxiv",
                SourcePaperRecord.source_id == "2302.01204v1",
            )
        )

        pdf_asset = session.scalar(select(PdfAssetRecord))

    assert source_paper is not None
    assert source_paper.canonical_paper_id is not None

    assert pdf_asset is not None
    assert pdf_asset.canonical_paper_id == source_paper.canonical_paper_id
    assert pdf_asset.source_paper_id == source_paper.id
    assert pdf_asset.source_url == "https://arxiv.org/pdf/2302.01204v1"
    assert pdf_asset.download_status == "pending"
    assert pdf_asset.local_file_path is None
    assert pdf_asset.sha256_checksum is None


def test_retrieval_does_not_register_pdf_asset_when_url_is_missing(
    tmp_path: Path,
) -> None:
    client = StaticPaperSourceClient(
        source_name="semantic_scholar",
        papers=[
            make_candidate(
                source="semantic_scholar",
                source_id="semantic-1",
                title="Example Paper Without PDF",
                open_access_pdf_url=None,
            )
        ],
    )

    service, session_factory = build_service(
        database_path=tmp_path / "academic_literature_rag.db",
        raw_directory=tmp_path / "raw",
        client=client,
    )

    result = service.search(
        query="example paper",
        limit=1,
    )

    assert result.run.status == "completed"

    with session_factory() as session:
        source_paper_count = session.scalar(select(func.count()).select_from(SourcePaperRecord))

        pdf_asset_count = session.scalar(select(func.count()).select_from(PdfAssetRecord))

    assert source_paper_count == 1
    assert pdf_asset_count == 0
