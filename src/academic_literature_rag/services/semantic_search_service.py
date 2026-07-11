from __future__ import annotations

from dataclasses import dataclass

from academic_literature_rag.models.chunk_embedding import ChunkEmbedding
from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)
from academic_literature_rag.models.text_chunk import TextChunk
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.retrieval.vector_math import cosine_similarity
from academic_literature_rag.services.embedding_client import EmbeddingClient


class SemanticSearchError(RuntimeError):
    """Raised when semantic search cannot be completed."""


@dataclass(frozen=True)
class ScoredChunk:
    """Internal scored chunk before ranking output conversion."""

    text_chunk: TextChunk
    chunk_embedding: ChunkEmbedding
    similarity_score: float


class SemanticSearchService:
    """Retrieves the most relevant text chunks for a user query."""

    def __init__(
        self,
        *,
        chunk_embedding_repository: ChunkEmbeddingRepository,
        text_chunk_repository: TextChunkRepository,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._chunk_embedding_repository = chunk_embedding_repository
        self._text_chunk_repository = text_chunk_repository
        self._embedding_client = embedding_client

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        """Return top-k semantically similar text chunks for a query."""

        normalized_query = query.strip()

        if not normalized_query:
            raise SemanticSearchError("Search query cannot be empty.")

        if top_k < 1:
            raise SemanticSearchError("top_k must be at least 1.")

        query_embedding = self._embedding_client.embed_text(
            normalized_query,
        )

        if query_embedding.model != self._embedding_client.model_name:
            raise SemanticSearchError("Query embedding model does not match client model.")

        chunk_embeddings = self._chunk_embedding_repository.list_by_model(
            self._embedding_client.model_name,
        )

        scored_chunks = [
            scored_chunk
            for chunk_embedding in chunk_embeddings
            if (
                scored_chunk := self._score_chunk_embedding(
                    query_vector=query_embedding.vector,
                    chunk_embedding=chunk_embedding,
                )
            )
            is not None
        ]

        ranked_chunks = sorted(
            scored_chunks,
            key=lambda scored_chunk: scored_chunk.similarity_score,
            reverse=True,
        )

        return [
            self._to_semantic_search_result(scored_chunk) for scored_chunk in ranked_chunks[:top_k]
        ]

    def _score_chunk_embedding(
        self,
        *,
        query_vector: list[float],
        chunk_embedding: ChunkEmbedding,
    ) -> ScoredChunk | None:
        """Score one chunk embedding against the query embedding."""

        text_chunk = self._text_chunk_repository.get(
            chunk_embedding.text_chunk_id,
        )

        if text_chunk is None:
            return None

        similarity_score = cosine_similarity(
            query_vector,
            chunk_embedding.embedding_vector,
        )

        return ScoredChunk(
            text_chunk=text_chunk,
            chunk_embedding=chunk_embedding,
            similarity_score=similarity_score,
        )

    @staticmethod
    def _to_semantic_search_result(
        scored_chunk: ScoredChunk,
    ) -> SemanticSearchResult:
        """Convert an internal scored chunk into a public search result."""

        return SemanticSearchResult(
            text_chunk_id=scored_chunk.text_chunk.text_chunk_id,
            pdf_asset_id=scored_chunk.text_chunk.pdf_asset_id,
            chunk_index=scored_chunk.text_chunk.chunk_index,
            start_page_number=scored_chunk.text_chunk.start_page_number,
            end_page_number=scored_chunk.text_chunk.end_page_number,
            text=scored_chunk.text_chunk.text,
            similarity_score=scored_chunk.similarity_score,
            embedding_model=scored_chunk.chunk_embedding.embedding_model,
        )
