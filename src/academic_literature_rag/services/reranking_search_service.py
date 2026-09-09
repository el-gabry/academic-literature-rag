"""Cross-encoder reranking service: a third retrieval stage that rescores
candidates from an existing retriever (dense or hybrid) for higher precision.

DESIGN RATIONALE:

1. This wraps an existing search service (SemanticSearchService or
   HybridSearchService) rather than replacing it. Reranking cannot invent
   relevant chunks that were never retrieved -- it can only reorder what's
   already in the candidate pool. So recall is bounded by the wrapped
   service's recall; only precision/NDCG can improve here.

2. The wrapped service is typed as a Protocol (structural typing) rather
   than a Union of concrete classes, so a third search service (e.g. a
   future SPLADE or ColBERT retriever) works here with zero code changes,
   as long as it exposes search(query, top_k=...) -> list[SemanticSearchResult].

3. candidate_pool_size (how many chunks to fetch before reranking) is kept
   deliberately larger than top_k (how many to return after reranking) --
   standard practice is retrieve ~20-50 broadly, rerank down to ~3-5.
   Passing candidate_pool_size == top_k would make reranking a no-op.

4. The cross-encoder is loaded lazily and cached on the instance, since
   model loading is expensive and should happen once, not per query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from academic_literature_rag.models.semantic_search_result import SemanticSearchResult


class RerankingError(RuntimeError):
    """Raised when reranking cannot be completed."""


class SearchService(Protocol):
    """Structural contract shared by SemanticSearchService and HybridSearchService."""

    def search(self, query: str, *, top_k: int = 5) -> list[SemanticSearchResult]:
        ...


@dataclass(frozen=True)
class _RerankedCandidate:
    """One candidate after cross-encoder rescoring."""

    result: SemanticSearchResult
    rerank_score: float


class RerankingSearchService:
    """Wraps a base search service with cross-encoder reranking.

    Drop-in alternative to SemanticSearchService/HybridSearchService: same
    search(query, top_k=...) call shape, same SemanticSearchResult return
    type, so RagPromptBuilder/RagAnswerService need no changes to consume it.
    """

    def __init__(
        self,
        *,
        base_search_service: SearchService,
        candidate_pool_size: int = 20,
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        if candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be at least 1.")

        self._base_search_service = base_search_service
        self._candidate_pool_size = candidate_pool_size
        self._cross_encoder_model_name = cross_encoder_model_name
        self._cross_encoder = None  # lazy-loaded on first search()

    def _get_cross_encoder(self):
        """Load the cross-encoder model on first use, then reuse it."""

        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RerankingError(
                    "sentence-transformers is required for reranking. "
                    "Install it with: pip install sentence-transformers"
                ) from error

            self._cross_encoder = CrossEncoder(self._cross_encoder_model_name)

        return self._cross_encoder

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        """Return top-k chunks after cross-encoder reranking of a base candidate pool."""

        normalized_query = query.strip()

        if not normalized_query:
            raise RerankingError("Search query cannot be empty.")

        if top_k < 1:
            raise RerankingError("top_k must be at least 1.")

        candidates = self._base_search_service.search(
            normalized_query,
            top_k=self._candidate_pool_size,
        )

        if not candidates:
            return []

        cross_encoder = self._get_cross_encoder()
        pairs = [(normalized_query, candidate.text) for candidate in candidates]
        scores = cross_encoder.predict(pairs)

        reranked = sorted(
            (
                _RerankedCandidate(result=candidate, rerank_score=float(score))
                for candidate, score in zip(candidates, scores, strict=True)
            ),
            key=lambda item: item.rerank_score,
            reverse=True,
        )

        return [
            self._to_semantic_search_result(item) for item in reranked[:top_k]
        ]

    @staticmethod
    def _to_semantic_search_result(
        candidate: _RerankedCandidate,
    ) -> SemanticSearchResult:
        """Replace the base retriever's score with the cross-encoder's rerank score.

        similarity_score is repurposed to carry the rerank score, keeping the
        SemanticSearchResult contract unchanged for downstream consumers.
        """

        base = candidate.result
        return SemanticSearchResult(
            text_chunk_id=base.text_chunk_id,
            pdf_asset_id=base.pdf_asset_id,
            chunk_index=base.chunk_index,
            start_page_number=base.start_page_number,
            end_page_number=base.end_page_number,
            text=base.text,
            similarity_score=candidate.rerank_score,
            embedding_model=f"{base.embedding_model}+rerank",
        )
