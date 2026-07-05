from __future__ import annotations

from datetime import UTC, datetime

from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
)
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)
from academic_literature_rag.repositories.source_paper_repository import (
    SourcePaperRepository,
)
from academic_literature_rag.storage.raw_response_store import (
    RawResponseStore,
)


class SemanticScholarRetrievalService:
    """Coordinates one persisted Semantic Scholar retrieval workflow."""

    def __init__(
        self,
        *,
        client: SemanticScholarClient,
        raw_response_store: RawResponseStore,
        search_run_repository: SearchRunRepository,
        source_paper_repository: SourcePaperRepository,
    ) -> None:
        self._client = client
        self._raw_response_store = raw_response_store
        self._search_run_repository = search_run_repository
        self._source_paper_repository = source_paper_repository

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        """Retrieve papers and persist the full retrieval workflow."""

        run = SearchRun(
            source="semantic_scholar",
            query=query,
        )

        # Save the attempt before calling the external API.
        self._search_run_repository.save(run)

        try:
            payload = self._client.fetch_search_payload(
                query=query,
                limit=limit,
            )

            raw_response_path = self._raw_response_store.save_json(
                source=run.source,
                run_id=run.run_id,
                payload=payload,
            )

            papers = self._client.map_search_payload(payload)

            self._source_paper_repository.save_for_run(
                run_id=run.run_id,
                papers=papers,
            )

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
