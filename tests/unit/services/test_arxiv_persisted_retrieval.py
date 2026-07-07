from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from academic_literature_rag.connectors.arxiv import (
    ArxivClient,
    ArxivRequestError,
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


FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "arxiv_search.xml"


def load_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def build_service(
    *,
    database_path: Path,
    raw_directory: Path,
    client: ArxivClient,
) -> tuple[
    PersistedRetrievalService,
    SearchRunRepository,
]:
    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    service = PersistedRetrievalService(
        client=client,
        raw_response_store=RawResponseStore(raw_directory),
        search_run_repository=SearchRunRepository(session_factory),
        source_paper_repository=SourcePaperRepository(session_factory),
    )

    return service, SearchRunRepository(session_factory)


def test_arxiv_workflow_persists_xml_and_papers(
    tmp_path: Path,
) -> None:
    fixture_xml = load_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/query"

        return httpx.Response(
            status_code=200,
            text=fixture_xml,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = ArxivClient(http_client=http_client)

        service, search_run_repository = build_service(
            database_path=tmp_path / "academic_literature_rag.db",
            raw_directory=tmp_path / "raw",
            client=client,
        )

        result = service.search(
            query="continuous glucose monitoring",
            limit=2,
        )

    persisted_run = search_run_repository.get(result.run.run_id)

    assert persisted_run is not None
    assert persisted_run.source == "arxiv"
    assert persisted_run.status == "completed"
    assert persisted_run.result_count == 2
    assert persisted_run.raw_response_path is not None

    raw_xml_path = Path(persisted_run.raw_response_path)

    assert raw_xml_path.exists()
    assert raw_xml_path.suffix == ".xml"
    assert raw_xml_path.read_text(encoding="utf-8") == fixture_xml

    assert len(result.papers) == 2
    assert result.papers[0].source_id == "2501.12345v2"
    assert result.papers[0].arxiv_id == "2501.12345"
    assert result.papers[1].source_id == "2502.67890v1"


def test_failed_arxiv_search_is_saved_in_sqlite(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            text="Service unavailable",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    engine = create_sqlite_engine(tmp_path / "academic_literature_rag.db")
    create_schema(engine)

    session_factory = create_session_factory(engine)

    with httpx.Client(transport=transport) as http_client:
        client = ArxivClient(http_client=http_client)

        service = PersistedRetrievalService(
            client=client,
            raw_response_store=RawResponseStore(tmp_path / "raw"),
            search_run_repository=SearchRunRepository(session_factory),
            source_paper_repository=SourcePaperRepository(session_factory),
        )

        with pytest.raises(ArxivRequestError):
            service.search(
                query="trigger arxiv error",
                limit=2,
            )

    with session_factory() as session:
        record = session.scalar(
            select(SearchRunRecord).where(
                SearchRunRecord.query == "trigger arxiv error",
            )
        )

    assert record is not None
    assert record.source == "arxiv"
    assert record.status == "failed"
    assert record.completed_at is not None
    assert record.error_message is not None
    assert "ArxivRequestError" in record.error_message
