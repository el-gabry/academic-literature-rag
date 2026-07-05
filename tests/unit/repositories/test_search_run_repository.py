from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)


def build_repository(database_path: Path) -> SearchRunRepository:
    engine = create_sqlite_engine(database_path)

    create_schema(engine)

    return SearchRunRepository(
        create_session_factory(engine),
    )


def test_saves_and_loads_search_run(tmp_path) -> None:
    repository = build_repository(
        tmp_path / "academic_literature_rag.db",
    )

    run = SearchRun(
        run_id=uuid4(),
        source="semantic_scholar",
        query="continuous glucose monitoring",
        started_at=datetime(2026, 7, 4, tzinfo=UTC),
        status="completed",
        result_count=2,
        raw_response_path="data/raw/semantic_scholar/example.json",
    )

    repository.save(run)

    loaded_run = repository.get(run.run_id)

    assert loaded_run is not None
    assert loaded_run.run_id == run.run_id
    assert loaded_run.source == "semantic_scholar"
    assert loaded_run.query == "continuous glucose monitoring"
    assert loaded_run.status == "completed"
    assert loaded_run.result_count == 2
    assert loaded_run.raw_response_path == ("data/raw/semantic_scholar/example.json")


def test_save_updates_existing_search_run(tmp_path) -> None:
    repository = build_repository(
        tmp_path / "academic_literature_rag.db",
    )

    run = SearchRun(
        run_id=uuid4(),
        source="semantic_scholar",
        query="glycemic variability",
    )

    repository.save(run)

    run.status = "completed"
    run.result_count = 5
    run.completed_at = datetime(2026, 7, 4, tzinfo=UTC)

    repository.save(run)

    loaded_run = repository.get(run.run_id)

    assert loaded_run is not None
    assert loaded_run.status == "completed"
    assert loaded_run.result_count == 5
    assert loaded_run.completed_at is not None


def test_returns_none_for_unknown_search_run(tmp_path) -> None:
    repository = build_repository(
        tmp_path / "academic_literature_rag.db",
    )

    missing_run = repository.get(uuid4())

    assert missing_run is None
