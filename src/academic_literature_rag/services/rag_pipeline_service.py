from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from academic_literature_rag.models.rag_answer import GroundedAnswer
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.services.chunk_embedding_service import (
    ChunkEmbeddingResult,
    ChunkEmbeddingService,
)
from academic_literature_rag.services.pending_pdf_download_service import (
    PendingPdfDownloadService,
)
from academic_literature_rag.services.persisted_retrieval_service import (
    PersistedRetrievalService,
)
from academic_literature_rag.services.pdf_text_extraction_service import (
    PdfTextExtractionService,
)
from academic_literature_rag.services.rag_answer_service import RagAnswerService
from academic_literature_rag.services.text_chunking_service import (
    TextChunkingService,
)


@dataclass(frozen=True)
class ProcessedPdfResult:
    """Summary of one PDF processing attempt."""

    pdf_asset_id: UUID
    source_url: str
    status: str
    page_count: int = 0
    chunk_count: int = 0
    error_message: str | None = None


@dataclass(frozen=True)
class RagPipelineIngestionResult:
    """Summary of one ingestion pipeline run."""

    retrieval_result: RetrievalResult
    pdf_results: list[ProcessedPdfResult]
    embedding_results: list[ChunkEmbeddingResult]


class RagPipelineError(RuntimeError):
    """Raised when the RAG pipeline cannot complete a requested operation."""


class RagPipelineService:
    """Orchestrates retrieval, PDF processing, embeddings, and answering."""

    def __init__(
        self,
        *,
        persisted_retrieval_service: PersistedRetrievalService,
        pending_pdf_download_service: PendingPdfDownloadService,
        pdf_text_extraction_service: PdfTextExtractionService,
        text_chunking_service: TextChunkingService,
        chunk_embedding_service: ChunkEmbeddingService,
        rag_answer_service: RagAnswerService | None = None,
    ) -> None:
        self._persisted_retrieval_service = persisted_retrieval_service
        self._pending_pdf_download_service = pending_pdf_download_service
        self._pdf_text_extraction_service = pdf_text_extraction_service
        self._text_chunking_service = text_chunking_service
        self._chunk_embedding_service = chunk_embedding_service
        self._rag_answer_service = rag_answer_service

    def ingest(
        self,
        *,
        query: str,
        retrieval_limit: int = 10,
        download_limit: int | None = None,
        embedding_limit: int | None = None,
    ) -> RagPipelineIngestionResult:
        """Run retrieval, pending PDF download, extraction, chunking, and embeddings."""

        normalized_query = query.strip()

        if not normalized_query:
            raise RagPipelineError("Pipeline query cannot be empty.")

        if retrieval_limit < 1:
            raise RagPipelineError("Retrieval limit must be at least 1.")

        retrieval_result = self._persisted_retrieval_service.search(
            query=normalized_query,
            limit=retrieval_limit,
        )

        download_results = self._pending_pdf_download_service.download_pending(
            limit=download_limit,
        )

        pdf_results = [
            self._process_downloaded_pdf(
                pdf_asset_id=download_result.pdf_asset_id,
                source_url=download_result.source_url,
                download_status=download_result.status,
                download_error_message=download_result.error_message,
            )
            for download_result in download_results
        ]

        embedding_results = self._chunk_embedding_service.embed_missing_chunks(
            limit=embedding_limit,
        )

        return RagPipelineIngestionResult(
            retrieval_result=retrieval_result,
            pdf_results=pdf_results,
            embedding_results=embedding_results,
        )

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> GroundedAnswer:
        """Generate a grounded answer from the current indexed knowledge base."""

        if self._rag_answer_service is None:
            raise RagPipelineError("RAG answer service is not configured.")

        return self._rag_answer_service.answer(
            question,
            top_k=top_k,
        )

    def _process_downloaded_pdf(
        self,
        *,
        pdf_asset_id: UUID,
        source_url: str,
        download_status: str,
        download_error_message: str | None,
    ) -> ProcessedPdfResult:
        """Extract and chunk one successfully downloaded PDF."""

        if download_status != "downloaded":
            return ProcessedPdfResult(
                pdf_asset_id=pdf_asset_id,
                source_url=source_url,
                status=download_status,
                error_message=download_error_message,
            )

        try:
            page_texts = self._pdf_text_extraction_service.extract(pdf_asset_id)
            chunks = self._text_chunking_service.chunk_pdf_asset(pdf_asset_id)

        except Exception as error:
            return ProcessedPdfResult(
                pdf_asset_id=pdf_asset_id,
                source_url=source_url,
                status="processing_failed",
                error_message=f"{type(error).__name__}: {error}",
            )

        return ProcessedPdfResult(
            pdf_asset_id=pdf_asset_id,
            source_url=source_url,
            status="processed",
            page_count=len(page_texts),
            chunk_count=len(chunks),
        )