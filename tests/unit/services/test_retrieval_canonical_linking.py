from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    CanonicalPaperRecord,
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
    """Small deterministic connector used for workflow integration tests."""

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


def make_candidate(
    *,
    source: str,
    source_id: str,
    title: str,
    doi: str,
) -> PaperCandidate:
    """Build one paper record for the integration test."""

    return PaperCandidate(
        source=source,
        source_id=source_id,
        title=title,
        landing_url=f"https://example.org/{source}/{source_id}",
        retrieved_at=datetime.now(UTC),
        authors=["A. Researcher"],
        publication_year=2026,
        doi=doi,
    )


def build_repositories(
    database_path: Path,
) -> tuple[
    sessionmaker[Session],
    SearchRunRepository,
    SourcePaperRepository,
    CanonicalPaperRepository,
]:
    """Create isolated SQLite repositories for one test."""

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = create_session_factory(engine)

    return (
        session_factory,
        SearchRunRepository(session_factory),
        SourcePaperRepository(session_factory),
        CanonicalPaperRepository(session_factory),
    )


def test_retrieval_automatically_links_same_doi_to_one_canonical_paper(
    tmp_path: Path,
) -> None:
    (
        session_factory,
        search_run_repository,
        source_paper_repository,
        canonical_paper_repository,
    ) = build_repositories(tmp_path / "academic_literature_rag.db")

    semantic_scholar_client = StaticPaperSourceClient(
        source_name="semantic_scholar",
        papers=[
            make_candidate(
                source="semantic_scholar",
                source_id="semantic-1",
                title="Early Warning Signals in Clinical Data",
                doi="https://doi.org/10.1000/SHARED.123",
            )
        ],
    )

    arxiv_client = StaticPaperSourceClient(
        source_name="arxiv",
        papers=[
            make_candidate(
                source="arxiv",
                source_id="2302.01204v1",
                title="Early-Warning Signals in Clinical Data",
                doi="doi:10.1000/shared.123",
            )
        ],
    )

    semantic_scholar_service = PersistedRetrievalService(
        client=semantic_scholar_client,
        raw_response_store=RawResponseStore(tmp_path / "raw"),
        search_run_repository=search_run_repository,
        source_paper_repository=source_paper_repository,
        canonical_paper_repository=canonical_paper_repository,
    )

    arxiv_service = PersistedRetrievalService(
        client=arxiv_client,
        raw_response_store=RawResponseStore(tmp_path / "raw"),
        search_run_repository=search_run_repository,
        source_paper_repository=source_paper_repository,
        canonical_paper_repository=canonical_paper_repository,
    )

    semantic_scholar_result = semantic_scholar_service.search(
        query="early warning signals",
        limit=1,
    )

    arxiv_result = arxiv_service.search(
        query="early warning signals",
        limit=1,
    )

    assert semantic_scholar_result.run.status == "completed"
    assert arxiv_result.run.status == "completed"

    with session_factory() as session:
        source_papers = session.scalars(
            select(SourcePaperRecord).order_by(SourcePaperRecord.source)
        ).all()

        canonical_paper_count = session.scalar(
            select(func.count()).select_from(CanonicalPaperRecord)
        )

    records_by_source = {record.source: record for record in source_papers}

    assert len(records_by_source) == 2
    assert canonical_paper_count == 1

    assert records_by_source["semantic_scholar"].canonical_paper_id is not None
    assert (
        records_by_source["semantic_scholar"].canonical_paper_id
        == records_by_source["arxiv"].canonical_paper_id
    )
