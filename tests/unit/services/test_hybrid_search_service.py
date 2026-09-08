from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)
from academic_literature_rag.services.hybrid_search_service import (
    HybridSearchError,
    HybridSearchService,
)
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchError,
)


@dataclass(frozen=True)
class FakeTextChunk:
    text_chunk_id: UUID
    pdf_asset_id: UUID
    chunk_index: int
    start_page_number: int
    end_page_number: int
    text: str


class FakeSemanticSearchService:
    def __init__(
        self,
        results: list[SemanticSearchResult] | None = None,
        *,
        should_raise: bool = False,
    ) -> None:
        self._results = results or []
        self._should_raise = should_raise
        self.search_calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        self.search_calls.append({"query": query, "top_k": top_k})

        if self._should_raise:
            raise SemanticSearchError("no embeddings")

        return self._results[:top_k]


class FakeTextChunkRepository:
    def __init__(self, chunks: list[FakeTextChunk]) -> None:
        self._chunks = chunks

    def list_all(self) -> list[FakeTextChunk]:
        return self._chunks


def make_chunk(chunk_id: UUID, text: str, *, chunk_index: int = 0) -> FakeTextChunk:
    return FakeTextChunk(
        text_chunk_id=chunk_id,
        pdf_asset_id=uuid4(),
        chunk_index=chunk_index,
        start_page_number=1,
        end_page_number=1,
        text=text,
    )


def make_dense_result(chunk: FakeTextChunk, score: float) -> SemanticSearchResult:
    return SemanticSearchResult(
        text_chunk_id=chunk.text_chunk_id,
        pdf_asset_id=chunk.pdf_asset_id,
        chunk_index=chunk.chunk_index,
        start_page_number=chunk.start_page_number,
        end_page_number=chunk.end_page_number,
        text=chunk.text,
        similarity_score=score,
        embedding_model="fake-embedding-model",
    )


def test_search_raises_when_index_not_built() -> None:
    chunk_a = make_chunk(uuid4(), "Attention mechanisms in transformers.")

    service = HybridSearchService(
        semantic_search_service=FakeSemanticSearchService([]),
        text_chunk_repository=FakeTextChunkRepository([chunk_a]),
    )

    with pytest.raises(HybridSearchError, match="index has not been built"):
        service.search("attention")


def test_refresh_index_reports_chunk_count() -> None:
    chunk_a = make_chunk(uuid4(), "Attention mechanisms in transformers.")
    chunk_b = make_chunk(uuid4(), "Recurrent neural networks.")

    service = HybridSearchService(
        semantic_search_service=FakeSemanticSearchService([]),
        text_chunk_repository=FakeTextChunkRepository([chunk_a, chunk_b]),
    )

    assert service.refresh_index() == 2


def test_search_surfaces_bm25_only_match_dense_search_missed() -> None:
    chunk_dense_hit = make_chunk(uuid4(), "Generic neural network background.")
    chunk_bm25_only = make_chunk(
        uuid4(),
        "Bayesian online changepoint detection for early warning signals.",
    )

    dense_service = FakeSemanticSearchService(
        [make_dense_result(chunk_dense_hit, score=0.9)]
    )

    service = HybridSearchService(
        semantic_search_service=dense_service,
        text_chunk_repository=FakeTextChunkRepository(
            [chunk_dense_hit, chunk_bm25_only]
        ),
    )

    service.refresh_index()

    results = service.search("Bayesian changepoint", top_k=5)

    result_ids = [result.text_chunk_id for result in results]
    assert chunk_bm25_only.text_chunk_id in result_ids


def test_search_falls_back_to_bm25_only_when_dense_search_fails() -> None:
    chunk_a = make_chunk(uuid4(), "Attention mechanisms in transformers.")

    service = HybridSearchService(
        semantic_search_service=FakeSemanticSearchService([], should_raise=True),
        text_chunk_repository=FakeTextChunkRepository([chunk_a]),
    )

    service.refresh_index()

    results = service.search("attention")

    assert len(results) == 1
    assert results[0].text_chunk_id == chunk_a.text_chunk_id
    assert results[0].embedding_model == "bm25-only"


def test_search_rejects_empty_query() -> None:
    chunk_a = make_chunk(uuid4(), "Some text.")

    service = HybridSearchService(
        semantic_search_service=FakeSemanticSearchService([]),
        text_chunk_repository=FakeTextChunkRepository([chunk_a]),
    )

    service.refresh_index()

    with pytest.raises(HybridSearchError, match="cannot be empty"):
        service.search("   ")


def test_search_rejects_invalid_top_k() -> None:
    chunk_a = make_chunk(uuid4(), "Some text.")

    service = HybridSearchService(
        semantic_search_service=FakeSemanticSearchService([]),
        text_chunk_repository=FakeTextChunkRepository([chunk_a]),
    )

    service.refresh_index()

    with pytest.raises(HybridSearchError, match="top_k"):
        service.search("some", top_k=0)


def test_constructor_rejects_invalid_rrf_k() -> None:
    with pytest.raises(ValueError, match="rrf_k"):
        HybridSearchService(
            semantic_search_service=FakeSemanticSearchService([]),
            text_chunk_repository=FakeTextChunkRepository([]),
            rrf_k=0,
        )


def test_constructor_rejects_invalid_candidate_pool_size() -> None:
    with pytest.raises(ValueError, match="candidate_pool_size"):
        HybridSearchService(
            semantic_search_service=FakeSemanticSearchService([]),
            text_chunk_repository=FakeTextChunkRepository([]),
            candidate_pool_size=0,
        )