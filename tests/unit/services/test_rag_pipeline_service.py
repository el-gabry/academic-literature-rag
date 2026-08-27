from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest

from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.models.rag_answer import AnswerCitation, GroundedAnswer
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.services.chunk_embedding_service import (
    ChunkEmbeddingResult,
    ChunkEmbeddingService,
)
from academic_literature_rag.services.pending_pdf_download_service import (
    PendingPdfDownloadResult,
    PendingPdfDownloadService,
)
from academic_literature_rag.services.pdf_text_extraction_service import (
    PdfTextExtractionService,
)
from academic_literature_rag.services.persisted_retrieval_service import (
    PersistedRetrievalService,
)
from academic_literature_rag.services.rag_answer_service import RagAnswerService
from academic_literature_rag.services.rag_pipeline_service import (
    RagPipelineError,
    RagPipelineService,
)
from academic_literature_rag.services.text_chunking_service import (
    TextChunkingService,
)


class FakePersistedRetrievalService:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.result = RetrievalResult(
            run=SearchRun(
                source="arxiv",
                query="retrieval augmented generation",
                status="completed",
                result_count=1,
            ),
            papers=[make_paper_candidate()],
        )

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> RetrievalResult:
        self.search_calls.append(
            {
                "query": query,
                "limit": limit,
            }
        )

        return self.result


class FakePendingPdfDownloadService:
    def __init__(
        self,
        download_results: list[PendingPdfDownloadResult],
    ) -> None:
        self.download_results = download_results
        self.download_pending_calls: list[dict[str, object]] = []

    def download_pending(
        self,
        *,
        limit: int | None = None,
    ) -> list[PendingPdfDownloadResult]:
        self.download_pending_calls.append(
            {
                "limit": limit,
            }
        )

        return self.download_results


class FakePdfTextExtractionService:
    def __init__(
        self,
        *,
        should_fail: bool = False,
    ) -> None:
        self.should_fail = should_fail
        self.extract_calls: list[UUID] = []

    def extract(
        self,
        pdf_asset_id: UUID,
    ) -> list[object]:
        self.extract_calls.append(pdf_asset_id)

        if self.should_fail:
            raise RuntimeError("text extraction failed")

        return [
            object(),
            object(),
        ]


class FakeTextChunkingService:
    def __init__(
        self,
        *,
        should_fail: bool = False,
    ) -> None:
        self.should_fail = should_fail
        self.chunk_pdf_asset_calls: list[UUID] = []

    def chunk_pdf_asset(
        self,
        pdf_asset_id: UUID,
    ) -> list[object]:
        self.chunk_pdf_asset_calls.append(pdf_asset_id)

        if self.should_fail:
            raise RuntimeError("chunking failed")

        return [
            object(),
            object(),
            object(),
        ]


class FakeChunkEmbeddingService:
    def __init__(
        self,
        embedding_results: list[ChunkEmbeddingResult] | None = None,
    ) -> None:
        self.embedding_results = embedding_results or []
        self.embed_missing_chunks_calls: list[dict[str, object]] = []

    def embed_missing_chunks(
        self,
        *,
        limit: int | None = None,
    ) -> list[ChunkEmbeddingResult]:
        self.embed_missing_chunks_calls.append(
            {
                "limit": limit,
            }
        )

        return self.embedding_results


class FakeRagAnswerService:
    def __init__(self) -> None:
        text_chunk_id = uuid4()
        pdf_asset_id = uuid4()

        self.answer_calls: list[dict[str, object]] = []
        self.answer_result = GroundedAnswer(
            question="What is RAG?",
            answer="RAG retrieves evidence before generating an answer.",
            citations=[
                AnswerCitation(
                    text_chunk_id=text_chunk_id,
                    pdf_asset_id=pdf_asset_id,
                    chunk_index=0,
                    start_page_number=1,
                    end_page_number=1,
                    similarity_score=0.95,
                    cited_text="Retrieval augmented generation retrieves evidence.",
                )
            ],
            generation_model="gpt-4.1-mini",
        )

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> GroundedAnswer:
        self.answer_calls.append(
            {
                "question": question,
                "top_k": top_k,
            }
        )

        return self.answer_result


def make_paper_candidate() -> PaperCandidate:
    return PaperCandidate(
        source="arxiv",
        source_id="1706.03762",
        title="Attention Is All You Need",
        landing_url="https://arxiv.org/abs/1706.03762",
        retrieved_at=datetime.now(UTC),
        abstract="Transformer paper.",
        authors=["Test Author"],
        publication_year=2017,
        venue="arXiv",
        doi=None,
        arxiv_id="1706.03762",
        open_access_pdf_url="https://arxiv.org/pdf/1706.03762",
        citation_count=None,
    )


def make_download_result(
    *,
    pdf_asset_id: UUID | None = None,
    status: str = "downloaded",
    error_message: str | None = None,
) -> PendingPdfDownloadResult:
    return PendingPdfDownloadResult(
        pdf_asset_id=pdf_asset_id or uuid4(),
        source_url="https://arxiv.org/pdf/1706.03762",
        status=status,
        error_message=error_message,
    )


def make_embedding_result() -> ChunkEmbeddingResult:
    return ChunkEmbeddingResult(
        text_chunk_id=uuid4(),
        status="embedded",
        embedding_model="text-embedding-3-small",
    )


def build_service(
    *,
    download_results: list[PendingPdfDownloadResult] | None = None,
    extraction_should_fail: bool = False,
    chunking_should_fail: bool = False,
    embedding_results: list[ChunkEmbeddingResult] | None = None,
    rag_answer_service: FakeRagAnswerService | None = None,
) -> tuple[
    RagPipelineService,
    FakePersistedRetrievalService,
    FakePendingPdfDownloadService,
    FakePdfTextExtractionService,
    FakeTextChunkingService,
    FakeChunkEmbeddingService,
]:
    persisted_retrieval_service = FakePersistedRetrievalService()
    pending_pdf_download_service = FakePendingPdfDownloadService(
        download_results=download_results or []
    )
    pdf_text_extraction_service = FakePdfTextExtractionService(
        should_fail=extraction_should_fail
    )
    text_chunking_service = FakeTextChunkingService(
        should_fail=chunking_should_fail
    )
    chunk_embedding_service = FakeChunkEmbeddingService(
        embedding_results=embedding_results
    )

    service = RagPipelineService(
        persisted_retrieval_service=cast(
            PersistedRetrievalService,
            persisted_retrieval_service,
        ),
        pending_pdf_download_service=cast(
            PendingPdfDownloadService,
            pending_pdf_download_service,
        ),
        pdf_text_extraction_service=cast(
            PdfTextExtractionService,
            pdf_text_extraction_service,
        ),
        text_chunking_service=cast(
            TextChunkingService,
            text_chunking_service,
        ),
        chunk_embedding_service=cast(
            ChunkEmbeddingService,
            chunk_embedding_service,
        ),
        rag_answer_service=cast(
            RagAnswerService | None,
            rag_answer_service,
        ),
    )

    return (
        service,
        persisted_retrieval_service,
        pending_pdf_download_service,
        pdf_text_extraction_service,
        text_chunking_service,
        chunk_embedding_service,
    )


def test_ingest_runs_retrieval_download_processing_and_embedding() -> None:
    pdf_asset_id = uuid4()
    embedding_result = make_embedding_result()

    (
        service,
        persisted_retrieval_service,
        pending_pdf_download_service,
        pdf_text_extraction_service,
        text_chunking_service,
        chunk_embedding_service,
    ) = build_service(
        download_results=[
            make_download_result(
                pdf_asset_id=pdf_asset_id,
                status="downloaded",
            )
        ],
        embedding_results=[embedding_result],
    )

    result = service.ingest(
        query="  retrieval augmented generation  ",
        retrieval_limit=3,
        download_limit=2,
        embedding_limit=1,
    )

    assert result.retrieval_result is persisted_retrieval_service.result
    assert result.embedding_results == [embedding_result]

    assert len(result.pdf_results) == 1
    assert result.pdf_results[0].pdf_asset_id == pdf_asset_id
    assert result.pdf_results[0].status == "processed"
    assert result.pdf_results[0].page_count == 2
    assert result.pdf_results[0].chunk_count == 3
    assert result.pdf_results[0].error_message is None

    assert persisted_retrieval_service.search_calls == [
        {
            "query": "retrieval augmented generation",
            "limit": 3,
        }
    ]
    assert pending_pdf_download_service.download_pending_calls == [
        {
            "limit": 2,
        }
    ]
    assert pdf_text_extraction_service.extract_calls == [pdf_asset_id]
    assert text_chunking_service.chunk_pdf_asset_calls == [pdf_asset_id]
    assert chunk_embedding_service.embed_missing_chunks_calls == [
        {
            "limit": 1,
        }
    ]


def test_ingest_does_not_process_failed_downloads() -> None:
    pdf_asset_id = uuid4()

    (
        service,
        _persisted_retrieval_service,
        _pending_pdf_download_service,
        pdf_text_extraction_service,
        text_chunking_service,
        _chunk_embedding_service,
    ) = build_service(
        download_results=[
            make_download_result(
                pdf_asset_id=pdf_asset_id,
                status="failed",
                error_message="not a pdf",
            )
        ],
    )

    result = service.ingest(
        query="retrieval augmented generation",
    )

    assert len(result.pdf_results) == 1
    assert result.pdf_results[0].pdf_asset_id == pdf_asset_id
    assert result.pdf_results[0].status == "failed"
    assert result.pdf_results[0].page_count == 0
    assert result.pdf_results[0].chunk_count == 0
    assert result.pdf_results[0].error_message == "not a pdf"

    assert pdf_text_extraction_service.extract_calls == []
    assert text_chunking_service.chunk_pdf_asset_calls == []


def test_ingest_records_processing_failure_when_extraction_fails() -> None:
    pdf_asset_id = uuid4()

    (service, *_rest) = build_service(
        download_results=[
            make_download_result(
                pdf_asset_id=pdf_asset_id,
                status="downloaded",
            )
        ],
        extraction_should_fail=True,
    )

    result = service.ingest(
        query="retrieval augmented generation",
    )

    assert len(result.pdf_results) == 1
    assert result.pdf_results[0].pdf_asset_id == pdf_asset_id
    assert result.pdf_results[0].status == "processing_failed"
    assert result.pdf_results[0].page_count == 0
    assert result.pdf_results[0].chunk_count == 0
    assert result.pdf_results[0].error_message == "RuntimeError: text extraction failed"


def test_ingest_records_processing_failure_when_chunking_fails() -> None:
    pdf_asset_id = uuid4()

    (service, *_rest) = build_service(
        download_results=[
            make_download_result(
                pdf_asset_id=pdf_asset_id,
                status="downloaded",
            )
        ],
        chunking_should_fail=True,
    )

    result = service.ingest(
        query="retrieval augmented generation",
    )

    assert len(result.pdf_results) == 1
    assert result.pdf_results[0].pdf_asset_id == pdf_asset_id
    assert result.pdf_results[0].status == "processing_failed"
    assert result.pdf_results[0].page_count == 0
    assert result.pdf_results[0].chunk_count == 0
    assert result.pdf_results[0].error_message == "RuntimeError: chunking failed"


def test_ingest_rejects_empty_query() -> None:
    (service, *_rest) = build_service()

    with pytest.raises(RagPipelineError, match="Pipeline query cannot be empty"):
        service.ingest(
            query="   ",
        )


def test_ingest_rejects_invalid_retrieval_limit() -> None:
    (service, *_rest) = build_service()

    with pytest.raises(RagPipelineError, match="Retrieval limit must be at least 1"):
        service.ingest(
            query="retrieval augmented generation",
            retrieval_limit=0,
        )


def test_answer_delegates_to_rag_answer_service() -> None:
    rag_answer_service = FakeRagAnswerService()
    (service, *_rest) = build_service(
        rag_answer_service=rag_answer_service,
    )

    result = service.answer(
        "What is RAG?",
        top_k=4,
    )

    assert result is rag_answer_service.answer_result
    assert rag_answer_service.answer_calls == [
        {
            "question": "What is RAG?",
            "top_k": 4,
        }
    ]


def test_answer_requires_configured_rag_answer_service() -> None:
    (service, *_rest) = build_service()

    with pytest.raises(RagPipelineError, match="RAG answer service is not configured"):
        service.answer(
            "What is RAG?",
        )