from __future__ import annotations

from dataclasses import dataclass

from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.retrieval.bm25 import Bm25Index, Bm25Match
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchError,
    SemanticSearchService,
)


class HybridSearchError(RuntimeError):
    """Raised when hybrid search cannot be completed."""


@dataclass(frozen=True)
class _RrfCandidate:
    """One candidate accumulated during Reciprocal Rank Fusion."""

    text_chunk_id: str
    rrf_score: float
    dense_result: SemanticSearchResult | None
    bm25_rank: int | None


class HybridSearchService:
    """Retrieves chunks using fused dense (embedding) and BM25 (lexical) ranking.

    Drop-in alternative to SemanticSearchService: same `search(query, top_k=...)`
    call shape, same SemanticSearchResult return type, so RagPromptBuilder and
    RagAnswerService need no changes to consume it.

    Fusion uses Reciprocal Rank Fusion (RRF), combining two ranked lists by
    rank position only, avoiding the need to normalize cosine similarity
    against BM25 scores, which live on different scales.
    """

    def __init__(
        self,
        *,
        semantic_search_service: SemanticSearchService,
        text_chunk_repository: TextChunkRepository,
        rrf_k: int = 60,
        candidate_pool_size: int = 50,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1.")

        self._semantic_search_service = semantic_search_service
        self._text_chunk_repository = text_chunk_repository
        self._rrf_k = rrf_k
        self._candidate_pool_size = candidate_pool_size
        self._bm25_index: Bm25Index | None = None
        self._chunk_lookup: dict[str, object] = {}

    def refresh_index(self) -> int:
        """Rebuild the BM25 index from the current text-chunk corpus.

        Call this once after ingestion (e.g. at the end of
        RagPipelineService.ingest) rather than on every search, since
        re-tokenizing the whole corpus per query is wasteful.
        """

        chunks = self._text_chunk_repository.list_all()

        index = Bm25Index()
        index.index(
            [(str(chunk.text_chunk_id), chunk.text) for chunk in chunks]
        )

        self._bm25_index = index
        self._chunk_lookup = {str(chunk.text_chunk_id): chunk for chunk in chunks}

        return index.size

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        """Return top-k chunks ranked by fused dense + BM25 relevance."""

        normalized_query = query.strip()

        if not normalized_query:
            raise HybridSearchError("Search query cannot be empty.")

        if top_k < 1:
            raise HybridSearchError("top_k must be at least 1.")

        if self._bm25_index is None:
            raise HybridSearchError(
                "BM25 index has not been built. Call refresh_index() after ingestion."
            )

        dense_results = self._run_dense_search(
            normalized_query,
            top_k=self._candidate_pool_size,
        )

        bm25_matches = self._bm25_index.search(
            normalized_query,
            top_k=self._candidate_pool_size,
        )

        fused_candidates = self._fuse(
            dense_results=dense_results,
            bm25_matches=bm25_matches,
        )

        return [
            self._to_semantic_search_result(candidate)
            for candidate in fused_candidates[:top_k]
        ]

    def _run_dense_search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SemanticSearchResult]:
        """Run dense search, treating an empty embedded corpus as zero results."""

        try:
            return self._semantic_search_service.search(
                query,
                top_k=top_k,
            )
        except SemanticSearchError:
            return []

    def _fuse(
        self,
        *,
        dense_results: list[SemanticSearchResult],
        bm25_matches: list[Bm25Match],
    ) -> list[_RrfCandidate]:
        """Combine dense and BM25 rankings using Reciprocal Rank Fusion."""

        rrf_scores: dict[str, float] = {}
        dense_by_id: dict[str, SemanticSearchResult] = {}
        bm25_rank_by_id: dict[str, int] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = str(result.text_chunk_id)
            dense_by_id[chunk_id] = result
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)

        for rank, match in enumerate(bm25_matches, start=1):
            chunk_id = match.document_id
            bm25_rank_by_id[chunk_id] = rank
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)

        ranked_ids = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            _RrfCandidate(
                text_chunk_id=chunk_id,
                rrf_score=score,
                dense_result=dense_by_id.get(chunk_id),
                bm25_rank=bm25_rank_by_id.get(chunk_id),
            )
            for chunk_id, score in ranked_ids
        ]

    def _to_semantic_search_result(
        self,
        candidate: _RrfCandidate,
    ) -> SemanticSearchResult:
        """Build a SemanticSearchResult for a fused candidate, filling gaps from storage."""

        if candidate.dense_result is not None:
            return SemanticSearchResult(
                text_chunk_id=candidate.dense_result.text_chunk_id,
                pdf_asset_id=candidate.dense_result.pdf_asset_id,
                chunk_index=candidate.dense_result.chunk_index,
                start_page_number=candidate.dense_result.start_page_number,
                end_page_number=candidate.dense_result.end_page_number,
                text=candidate.dense_result.text,
                similarity_score=candidate.rrf_score,
                embedding_model=candidate.dense_result.embedding_model,
            )

        chunk = self._chunk_lookup.get(candidate.text_chunk_id)

        if chunk is None:
            raise HybridSearchError(
                f"BM25-only candidate has no matching chunk in storage: "
                f"{candidate.text_chunk_id}"
            )

        return SemanticSearchResult(
            text_chunk_id=chunk.text_chunk_id,
            pdf_asset_id=chunk.pdf_asset_id,
            chunk_index=chunk.chunk_index,
            start_page_number=chunk.start_page_number,
            end_page_number=chunk.end_page_number,
            text=chunk.text,
            similarity_score=candidate.rrf_score,
            embedding_model="bm25-only",
        )