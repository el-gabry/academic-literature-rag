from __future__ import annotations

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
from academic_literature_rag.services.generation_client import (
    GenerationResponse,
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
    """Same deterministic embedding client used by the pipeline smoke script."""

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


class SmokeGenerationClient:
    """Small deterministic generation client for local smoke runs."""

    @property
    def model_name(
        self,
    ) -> str:
        """Return the smoke generation model name."""

        return "smoke-deterministic-generation"

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        """Generate a simple grounded answer from the first retrieved evidence block."""

        return GenerationResponse(
            model=self.model_name,
            text=self._build_answer_from_prompt(prompt),
        )

    @staticmethod
    def _build_answer_from_prompt(
        prompt: str,
    ) -> str:
        """Build a short answer using only evidence text from the prompt."""

        text_marker = "Text:\n"

        if text_marker not in prompt:
            return "The evidence is insufficient to answer the question."

        first_evidence_text = prompt.split(text_marker, 1)[1]
        first_evidence_text = first_evidence_text.split("\n\n[Evidence", 1)[0]
        first_evidence_text = first_evidence_text.split("\n\nAnswer:", 1)[0]
        first_evidence_text = " ".join(first_evidence_text.split())

        if not first_evidence_text:
            return "The evidence is insufficient to answer the question."

        return (
            "Based on the retrieved evidence, the answer is grounded in this passage: "
            f"{first_evidence_text[:700]}"
        )


def main() -> None:
    """Run one local grounded-answer smoke test."""

    database_path = Path("data/db/dev.db")
    question = "What is attention in transformers?"
    top_k = 3

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
        generation_client=SmokeGenerationClient(),
    )

    answer = rag_answer_service.answer(
        question,
        top_k=top_k,
    )

    print_answer(answer)


def ensure_smoke_embeddings_exist(
    *,
    chunk_embedding_repository: ChunkEmbeddingRepository,
    embedding_model: str,
) -> None:
    """Fail with a clear message if the ingestion smoke has not run yet."""

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

    print("RAG answer smoke run completed.")
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