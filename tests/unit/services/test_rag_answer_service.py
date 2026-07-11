from __future__ import annotations

from uuid import uuid4

import pytest

from academic_literature_rag.models.rag_answer import GroundedAnswer
from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)
from academic_literature_rag.services.generation_client import (
    GenerationResponse,
)
from academic_literature_rag.services.rag_answer_service import (
    RagAnswerError,
    RagAnswerService,
)
from academic_literature_rag.services.rag_prompt_builder import (
    RagPromptBuilder,
)


class FakeSemanticSearchService:
    """Fake semantic search service for RAG answer tests."""

    def __init__(
        self,
        retrieved_chunks: list[SemanticSearchResult],
    ) -> None:
        self._retrieved_chunks = retrieved_chunks
        self.search_calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        self.search_calls.append(
            (
                query,
                top_k,
            )
        )

        return self._retrieved_chunks


class FakeGenerationClient:
    """Fake generation client for RAG answer tests."""

    def __init__(
        self,
        *,
        model_name: str = "fake-generation-model",
        response_text: str = "Self-attention allows tokens to attend to each other.",
        response_model: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._response_text = response_text
        self._response_model = response_model
        self.prompts: list[str] = []

    @property
    def model_name(
        self,
    ) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        self.prompts.append(prompt)

        return GenerationResponse(
            model=self._response_model or self._model_name,
            text=self._response_text,
        )


def build_search_result(
    *,
    chunk_index: int = 0,
    text: str = "The Transformer uses self-attention mechanisms.",
    similarity_score: float = 0.95,
) -> SemanticSearchResult:
    """Create one retrieved chunk for RAG answer tests."""

    return SemanticSearchResult(
        text_chunk_id=uuid4(),
        pdf_asset_id=uuid4(),
        chunk_index=chunk_index,
        start_page_number=2,
        end_page_number=3,
        text=text,
        similarity_score=similarity_score,
        embedding_model="fake-embedding-model",
    )


def build_service(
    *,
    retrieved_chunks: list[SemanticSearchResult],
    generation_client: FakeGenerationClient | None = None,
) -> tuple[
    RagAnswerService,
    FakeSemanticSearchService,
    FakeGenerationClient,
]:
    """Create the RAG answer service with fake dependencies."""

    semantic_search_service = FakeSemanticSearchService(
        retrieved_chunks=retrieved_chunks,
    )

    resolved_generation_client = generation_client or FakeGenerationClient()

    service = RagAnswerService(
        semantic_search_service=semantic_search_service,  # type: ignore[arg-type]
        rag_prompt_builder=RagPromptBuilder(),
        generation_client=resolved_generation_client,
    )

    return (
        service,
        semantic_search_service,
        resolved_generation_client,
    )


def test_answer_generates_grounded_answer_with_citations() -> None:
    retrieved_chunk = build_search_result()

    service, semantic_search_service, generation_client = build_service(
        retrieved_chunks=[retrieved_chunk],
    )

    grounded_answer = service.answer(
        "What is self-attention?",
        top_k=3,
    )

    assert isinstance(grounded_answer, GroundedAnswer)
    assert grounded_answer.question == "What is self-attention?"
    assert grounded_answer.answer == "Self-attention allows tokens to attend to each other."
    assert grounded_answer.generation_model == "fake-generation-model"

    assert len(grounded_answer.citations) == 1

    citation = grounded_answer.citations[0]

    assert citation.text_chunk_id == retrieved_chunk.text_chunk_id
    assert citation.pdf_asset_id == retrieved_chunk.pdf_asset_id
    assert citation.chunk_index == retrieved_chunk.chunk_index
    assert citation.start_page_number == retrieved_chunk.start_page_number
    assert citation.end_page_number == retrieved_chunk.end_page_number
    assert citation.similarity_score == retrieved_chunk.similarity_score
    assert citation.cited_text == retrieved_chunk.text

    assert semantic_search_service.search_calls == [
        (
            "What is self-attention?",
            3,
        )
    ]

    assert len(generation_client.prompts) == 1
    assert "What is self-attention?" in generation_client.prompts[0]
    assert retrieved_chunk.text in generation_client.prompts[0]


def test_answer_strips_question_before_search_and_generation() -> None:
    retrieved_chunk = build_search_result()

    service, semantic_search_service, generation_client = build_service(
        retrieved_chunks=[retrieved_chunk],
    )

    grounded_answer = service.answer(
        "   What is the Transformer?   ",
    )

    assert grounded_answer.question == "What is the Transformer?"

    assert semantic_search_service.search_calls == [
        (
            "What is the Transformer?",
            5,
        )
    ]

    assert "Question:\nWhat is the Transformer?" in generation_client.prompts[0]
    assert "   What is the Transformer?   " not in generation_client.prompts[0]


def test_answer_includes_multiple_citations() -> None:
    first_chunk = build_search_result(
        chunk_index=0,
        text="First evidence chunk.",
        similarity_score=0.91,
    )

    second_chunk = build_search_result(
        chunk_index=1,
        text="Second evidence chunk.",
        similarity_score=0.82,
    )

    service, _semantic_search_service, generation_client = build_service(
        retrieved_chunks=[
            first_chunk,
            second_chunk,
        ],
    )

    grounded_answer = service.answer("Explain the model.")

    assert len(grounded_answer.citations) == 2

    assert [citation.text_chunk_id for citation in grounded_answer.citations] == [
        first_chunk.text_chunk_id,
        second_chunk.text_chunk_id,
    ]

    assert "First evidence chunk." in generation_client.prompts[0]
    assert "Second evidence chunk." in generation_client.prompts[0]


def test_answer_rejects_empty_question() -> None:
    service, _semantic_search_service, _generation_client = build_service(
        retrieved_chunks=[build_search_result()],
    )

    with pytest.raises(RagAnswerError, match="Question cannot be empty"):
        service.answer("   ")


def test_answer_rejects_when_no_chunks_are_retrieved() -> None:
    service, _semantic_search_service, _generation_client = build_service(
        retrieved_chunks=[],
    )

    with pytest.raises(
        RagAnswerError,
        match="No relevant chunks were retrieved",
    ):
        service.answer("What is attention?")


def test_answer_rejects_generation_model_mismatch() -> None:
    service, _semantic_search_service, _generation_client = build_service(
        retrieved_chunks=[build_search_result()],
        generation_client=FakeGenerationClient(
            model_name="expected-model",
            response_model="different-model",
        ),
    )

    with pytest.raises(RagAnswerError, match="does not match"):
        service.answer("What is attention?")


def test_answer_rejects_empty_generation_response() -> None:
    service, _semantic_search_service, _generation_client = build_service(
        retrieved_chunks=[build_search_result()],
        generation_client=FakeGenerationClient(
            response_text="   ",
        ),
    )

    with pytest.raises(RagAnswerError, match="empty answer"):
        service.answer("What is attention?")
