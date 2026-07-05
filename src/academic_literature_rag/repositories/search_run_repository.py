from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.database.models import SearchRunRecord
from academic_literature_rag.models.search_run import SearchRun


class SearchRunRepository:
    """Persists and retrieves SearchRun records."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, run: SearchRun) -> None:
        """Create or update one search run."""

        with self._session_factory.begin() as session:
            record = session.get(
                SearchRunRecord,
                str(run.run_id),
            )

            if record is None:
                session.add(
                    SearchRunRecord(
                        run_id=str(run.run_id),
                        source=run.source,
                        query=run.query,
                        started_at=run.started_at,
                        completed_at=run.completed_at,
                        status=run.status,
                        result_count=run.result_count,
                        raw_response_path=run.raw_response_path,
                        error_message=run.error_message,
                    )
                )
                return

            record.source = run.source
            record.query = run.query
            record.started_at = run.started_at
            record.completed_at = run.completed_at
            record.status = run.status
            record.result_count = run.result_count
            record.raw_response_path = run.raw_response_path
            record.error_message = run.error_message

    def get(self, run_id: UUID) -> SearchRun | None:
        """Return one search run, or None when it does not exist."""

        with self._session_factory() as session:
            record = session.get(
                SearchRunRecord,
                str(run_id),
            )

            if record is None:
                return None

            return SearchRun(
                run_id=UUID(record.run_id),
                source=record.source,
                query=record.query,
                started_at=record.started_at,
                completed_at=record.completed_at,
                status=record.status,
                result_count=record.result_count,
                raw_response_path=record.raw_response_path,
                error_message=record.error_message,
            )
