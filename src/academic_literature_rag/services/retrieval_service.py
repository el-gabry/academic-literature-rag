from __future__ import annotations

from datetime import UTC, datetime


from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
)
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.storage.raw_response_store import (
    RawResponseStore,
)


class SemanticScholarRetrievalService:
    """Coordinates retrieval, raw-response preservation, and normalization."""

    def __init__(
        self,
        *,
        client: SemanticScholarClient,
        raw_response_store: RawResponseStore,
    ) -> None:
        self._client = client
        self._raw_response_store = raw_response_store

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        run = SearchRun(
            source="semantic_scholar",
            query=query,
        )

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

        run.status = "completed"
        run.completed_at = datetime.now(UTC)
        run.result_count = len(papers)
        run.raw_response_path = str(raw_response_path)

        return RetrievalResult(
            run=run,
            papers=papers,
        )
