from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import (
    SearchRunPaperRecord,
    SourcePaperRecord,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)
from academic_literature_rag.repositories.source_paper_repository import (
    SourcePaperRepository,
)


def build_repositories(
    database_path: Path,
) -> tuple[
    SearchRunRepository,
    SourcePaperRepository,
    sessionmaker[Session],
]:
    engine = create_sqlite_engine(database_path)

    create_schema(engine)

    session_factory = create_session_factory(engine)

    return (
        SearchRunRepository(session_factory),
        SourcePaperRepository(session_factory),
        session_factory,
    )


def build_candidate() -> PaperCandidate:
    return PaperCandidate(
        source="semantic_scholar",
        source_id="paper-001",
        title="Continuous Glucose Monitoring in Older Adults",
        landing_url="https://example.com/paper-001",
        retrieved_at=datetime(2026, 7, 4, tzinfo=UTC),
        authors=["Jane Smith", "John Doe"],
        publication_year=2025,
        doi="10.1000/example.001",
        citation_count=12,
    )


def test_saves_and_loads_papers_for_search_run(tmp_path) -> None:
    run_repository, paper_repository, _ = build_repositories(
        tmp_path / "academic_literature_rag.db",
    )

    run = SearchRun(
        run_id=uuid4(),
        source="semantic_scholar",
        query="continuous glucose monitoring",
    )

    run_repository.save(run)

    paper_repository.save_for_run(
        run_id=run.run_id,
        papers=[build_candidate()],
    )

    loaded_papers = paper_repository.list_for_run(run.run_id)

    assert len(loaded_papers) == 1
    assert loaded_papers[0].source_id == "paper-001"
    assert loaded_papers[0].authors == ["Jane Smith", "John Doe"]
    assert loaded_papers[0].doi == "10.1000/example.001"


def test_reuses_one_source_paper_across_multiple_search_runs(tmp_path) -> None:
    run_repository, paper_repository, session_factory = build_repositories(
        tmp_path / "academic_literature_rag.db",
    )

    first_run = SearchRun(
        run_id=uuid4(),
        source="semantic_scholar",
        query="continuous glucose monitoring",
    )
    second_run = SearchRun(
        run_id=uuid4(),
        source="semantic_scholar",
        query="glycemic variability",
    )

    run_repository.save(first_run)
    run_repository.save(second_run)

    paper_repository.save_for_run(
        run_id=first_run.run_id,
        papers=[build_candidate()],
    )
    paper_repository.save_for_run(
        run_id=second_run.run_id,
        papers=[build_candidate()],
    )

    with session_factory() as session:
        source_paper_count = session.scalar(select(func.count()).select_from(SourcePaperRecord))
        run_link_count = session.scalar(select(func.count()).select_from(SearchRunPaperRecord))

    assert source_paper_count == 1
    assert run_link_count == 2
