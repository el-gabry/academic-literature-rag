from __future__ import annotations

import os
from pathlib import Path

from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.rag_answer import GroundedAnswer
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.services.embedding_client import EmbeddingResponse
from academic_literature_rag.services.openai_generation_client import (
    OpenAIGenerationClient,
)
from academic_literature_rag.services.rag_answer_service import (
    RagAnswerService,
)
from academic_literature_rag.services.rag_prompt_builder import (
    RagPromptBuilder,
)
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchService,
)


class SmokeEmbeddingClient:
    """Deterministic embedding client matching smoke_rag_pipeline.py."""

    @property
    def model_name(
        self,
    ) -> str:
        """Return the smoke embedding model name."""

        return "smoke-deterministic-embedding"

    def embed_text(
        self,
        text: str,
    ) -> EmbeddingResponse:
        """Return a deterministic vector without calling an external API."""

        return EmbeddingResponse(
            model=self.model_name,
            vector=self._build_vector(text),
        )

    @staticmethod
    def _build_vector(
        text: str,
    ) -> list[float]:
        """Create a stable tiny vector from text content."""

        cleaned_text = text.strip()

        if not cleaned_text:
            return [0.0] * 8

        buckets = [0.0] * 8

        for index, character in enumerate(cleaned_text):
            buckets[index % len(buckets)] += float(ord(character) % 101)

        total = sum(buckets) or 1.0

        return [value / total for value in buckets]


def main() -> None:
    """Run one OpenAI-backed grounded-answer smoke test."""

    ensure_openai_api_key_exists()

    database_path = Path("data/db/dev.db")
    question = "What is attention in transformers?"
    top_k = 3
    model_name = os.getenv(
        "OPENAI_GENERATION_MODEL",
        "gpt-4o-mini",
    )

    engine = create_sqlite_engine(database_path)
    create_schema(engine)
    session_factory = create_session_factory(engine)

    chunk_embedding_repository = ChunkEmbeddingRepository(session_factory)
    text_chunk_repository = TextChunkRepository(session_factory)
    embedding_client = SmokeEmbeddingClient()

    ensure_smoke_embeddings_exist(
        chunk_embedding_repository=chunk_embedding_repository,
        embedding_model=embedding_client.model_name,
    )

    semantic_search_service = SemanticSearchService(
        chunk_embedding_repository=chunk_embedding_repository,
        text_chunk_repository=text_chunk_repository,
        embedding_client=embedding_client,
    )

    rag_answer_service = RagAnswerService(
        semantic_search_service=semantic_search_service,
        rag_prompt_builder=RagPromptBuilder(),
        generation_client=OpenAIGenerationClient(
            model_name=model_name,
        ),
    )

    answer = rag_answer_service.answer(
        question,
        top_k=top_k,
    )

    print_answer(answer)


def ensure_openai_api_key_exists() -> None:
    """Fail early if OPENAI_API_KEY is missing."""

    if os.getenv("OPENAI_API_KEY"):
        return

    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it before running this script:\n"
        "export OPENAI_API_KEY='your-key-here'"
    )


def ensure_smoke_embeddings_exist(
    *,
    chunk_embedding_repository: ChunkEmbeddingRepository,
    embedding_model: str,
) -> None:
    """Fail clearly if the ingestion smoke has not run yet."""

    embeddings = chunk_embedding_repository.list_by_model(embedding_model)

    if embeddings:
        return

    raise RuntimeError(
        "No smoke embeddings found. Run this first:\n"
        "uv run python scripts/smoke_rag_pipeline.py"
    )


def print_answer(
    answer: GroundedAnswer,
) -> None:
    """Print a readable grounded answer summary."""

    print("OpenAI RAG answer smoke run completed.")
    print()
    print("Question")
    print(f"- {answer.question}")

    print()
    print("Answer")
    print(answer.answer)

    print()
    print("Generation")
    print(f"- Model: {answer.generation_model}")
    print(f"- Created at: {answer.created_at}")

    print()
    print("Citations")
    for index, citation in enumerate(answer.citations, start=1):
        print(f"{index}. Chunk ID: {citation.text_chunk_id}")
        print(f"   PDF asset ID: {citation.pdf_asset_id}")
        print(f"   Chunk index: {citation.chunk_index}")
        print(f"   Pages: {citation.start_page_number}-{citation.end_page_number}")
        print(f"   Similarity: {citation.similarity_score:.4f}")
        print(f"   Text: {' '.join(citation.cited_text.split())[:500]}")


if __name__ == "__main__":
    main()