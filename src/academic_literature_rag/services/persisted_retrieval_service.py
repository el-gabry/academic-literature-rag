from __future__ import annotations

from datetime import UTC, datetime

from academic_literature_rag.connectors.protocols import PaperSourceClient
from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.models.search_run import SearchRun
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
from academic_literature_rag.storage.raw_response_store import RawResponseStore


class PersistedRetrievalService:
    """Coordinates retrieval, persistence, canonicalization, and PDF registration."""

    def __init__(
        self,
        *,
        client: PaperSourceClient,
        raw_response_store: RawResponseStore,
        search_run_repository: SearchRunRepository,
        source_paper_repository: SourcePaperRepository,
        canonical_paper_repository: CanonicalPaperRepository,
        pdf_asset_repository: PdfAssetRepository,
    ) -> None:
        self._client = client
        self._raw_response_store = raw_response_store
        self._search_run_repository = search_run_repository
        self._source_paper_repository = source_paper_repository
        self._canonical_paper_repository = canonical_paper_repository
        self._pdf_asset_repository = pdf_asset_repository

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        """Retrieve, persist, canonicalize, and register PDF candidates."""

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

            self._register_canonical_papers_and_pdf_assets(
                papers=papers,
                source_paper_ids=source_paper_ids,
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

    def _register_canonical_papers_and_pdf_assets(
        self,
        *,
        papers: list[PaperCandidate],
        source_paper_ids: list,
    ) -> None:
        """Link source papers to canonical records and register PDF URLs."""

        for paper, source_paper_id in zip(
            papers,
            source_paper_ids,
            strict=True,
        ):
            canonical_paper = self._canonical_paper_repository.link_or_create_for_source_paper(
                source_paper_id
            )

            if paper.open_access_pdf_url is None:
                continue

            self._pdf_asset_repository.create_or_get_pending(
                canonical_paper_id=canonical_paper.canonical_paper_id,
                source_paper_id=source_paper_id,
                source_url=paper.open_access_pdf_url,
            )
