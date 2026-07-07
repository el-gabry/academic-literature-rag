from __future__ import annotations

from datetime import UTC, datetime

from academic_literature_rag.connectors.protocols import PaperSourceClient
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.repositories.canonical_paper_repository import (
    CanonicalPaperRepository,
)
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)
from academic_literature_rag.repositories.source_paper_repository import (
    SourcePaperRepository,
)
from academic_literature_rag.storage.raw_response_store import RawResponseStore


class PersistedRetrievalService:
    """Coordinates a persisted retrieval workflow for any paper source."""

    def __init__(
        self,
        *,
        client: PaperSourceClient,
        raw_response_store: RawResponseStore,
        search_run_repository: SearchRunRepository,
        source_paper_repository: SourcePaperRepository,
        canonical_paper_repository: CanonicalPaperRepository,
    ) -> None:
        self._client = client
        self._raw_response_store = raw_response_store
        self._search_run_repository = search_run_repository
        self._source_paper_repository = source_paper_repository
        self._canonical_paper_repository = canonical_paper_repository

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        """Retrieve, persist, and safely canonicalize paper records."""

        run = SearchRun(
            source=self._client.source_name,
            query=query,
        )

        self._search_run_repository.save(run)

        try:
            raw_response = self._client.fetch_raw_response(
                query=query,
                limit=limit,
            )

            raw_response_path = self._raw_response_store.save_text(
                source=run.source,
                run_id=run.run_id,
                content=raw_response,
                extension=self._client.raw_extension,
            )

            papers = self._client.map_raw_response(raw_response)

            source_paper_ids = self._source_paper_repository.save_for_run(
                run_id=run.run_id,
                papers=papers,
            )

            for source_paper_id in source_paper_ids:
                self._canonical_paper_repository.link_or_create_for_source_paper(source_paper_id)

            run.status = "completed"
            run.completed_at = datetime.now(UTC)
            run.result_count = len(papers)
            run.raw_response_path = str(raw_response_path)

            self._search_run_repository.save(run)

            return RetrievalResult(
                run=run,
                papers=papers,
            )

        except Exception as error:
            run.status = "failed"
            run.completed_at = datetime.now(UTC)
            run.error_message = f"{type(error).__name__}: {error}"

            self._search_run_repository.save(run)

            raise
