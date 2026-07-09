from __future__ import annotations

import json
from pathlib import Path
from academic_literature_rag.repositories.canonical_paper_repository import (
    CanonicalPaperRepository,
)

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarRequestError,
)
from academic_literature_rag.database.models import SearchRunRecord
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
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
from academic_literature_rag.storage.raw_response_store import (
    RawResponseStore,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)


FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "semantic_scholar_search.json"


def load_fixture() -> dict[str, object]:
    """Load the fixed Semantic Scholar API response used in tests."""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def build_service(
    *,
    database_path: Path,
    raw_directory: Path,
    client: SemanticScholarClient,
) -> tuple[
    PersistedRetrievalService,
    SearchRunRepository,
    SourcePaperRepository,
    sessionmaker[Session],
]:
    """Build a complete retrieval service with a temporary SQLite database."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    search_run_repository = SearchRunRepository(session_factory)
    source_paper_repository = SourcePaperRepository(session_factory)

    service = PersistedRetrievalService(
        client=client,
        canonical_paper_repository=CanonicalPaperRepository(session_factory),
        pdf_asset_repository=PdfAssetRepository(session_factory),
        raw_response_store=RawResponseStore(raw_directory),
        search_run_repository=search_run_repository,
        source_paper_repository=source_paper_repository,
    )

    return (
        service,
        search_run_repository,
        source_paper_repository,
        session_factory,
    )


def test_successful_search_persists_complete_workflow(tmp_path: Path) -> None:
    """A successful search saves raw JSON, run metadata, and papers."""

    fixture_payload = load_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=fixture_payload,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = SemanticScholarClient(http_client=http_client)

        (
            service,
            search_run_repository,
            source_paper_repository,
            _,
        ) = build_service(
            database_path=tmp_path / "academic_literature_rag.db",
            raw_directory=tmp_path / "raw",
            client=client,
        )

        result = service.search(
            query="continuous glucose monitoring",
            limit=2,
        )

    persisted_run = search_run_repository.get(result.run.run_id)

    persisted_papers = source_paper_repository.list_for_run(
        result.run.run_id,
    )

    assert persisted_run is not None
    assert persisted_run.status == "completed"
    assert persisted_run.result_count == 2
    assert persisted_run.raw_response_path is not None

    raw_response_path = Path(persisted_run.raw_response_path)

    assert raw_response_path.exists()

    saved_payload = json.loads(raw_response_path.read_text(encoding="utf-8"))

    assert saved_payload == fixture_payload

    assert len(persisted_papers) == 2
    assert persisted_papers[0].source_id == "paper-001"
    assert persisted_papers[1].source_id == "paper-002"


def test_failed_search_is_saved_with_error_message(tmp_path: Path) -> None:
    """A failed API request is recorded in SQLite with failure details."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={"message": "Too many requests"},
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = SemanticScholarClient(http_client=http_client)

        (
            service,
            _,
            _,
            session_factory,
        ) = build_service(
            database_path=tmp_path / "academic_literature_rag.db",
            raw_directory=tmp_path / "raw",
            client=client,
        )

        with pytest.raises(SemanticScholarRequestError):
            service.search(
                query="trigger a rate-limit error",
                limit=2,
            )

    with session_factory() as session:
        record = session.scalar(
            select(SearchRunRecord).where(
                SearchRunRecord.query == "trigger a rate-limit error",
            )
        )

    assert record is not None
    assert record.status == "failed"
    assert record.completed_at is not None
    assert record.error_message is not None
    assert "SemanticScholarRequestError" in record.error_message
