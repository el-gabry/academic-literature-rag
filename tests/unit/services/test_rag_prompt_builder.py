from __future__ import annotations

from uuid import uuid4

import pytest

from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)
from academic_literature_rag.services.rag_prompt_builder import (
    RagPromptBuilder,
    RagPromptBuilderError,
)


def build_search_result(
    *,
    chunk_index: int = 0,
    text: str = "The Transformer uses self-attention mechanisms.",
    similarity_score: float = 0.95,
) -> SemanticSearchResult:
    """Create one semantic search result for prompt-builder tests."""

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


def test_build_prompt_includes_question_and_evidence() -> None:
    builder = RagPromptBuilder()

    retrieved_chunk = build_search_result()

    prompt = builder.build_prompt(
        question="What is self-attention?",
        retrieved_chunks=[retrieved_chunk],
    )

    assert "You are an academic research assistant." in prompt
    assert "Answer the question using only the provided evidence." in prompt
    assert "What is self-attention?" in prompt
    assert "[Evidence 1]" in prompt
    assert f"Chunk ID: {retrieved_chunk.text_chunk_id}" in prompt
    assert f"PDF Asset ID: {retrieved_chunk.pdf_asset_id}" in prompt
    assert "Chunk Index: 0" in prompt
    assert "Pages: 2-3" in prompt
    assert "Similarity Score: 0.9500" in prompt
    assert "The Transformer uses self-attention mechanisms." in prompt
    assert prompt.endswith("Answer:")


def test_build_prompt_includes_multiple_evidence_items() -> None:
    builder = RagPromptBuilder()

    first_chunk = build_search_result(
        chunk_index=0,
        text="First retrieved evidence.",
        similarity_score=0.91,
    )

    second_chunk = build_search_result(
        chunk_index=1,
        text="Second retrieved evidence.",
        similarity_score=0.82,
    )

    prompt = builder.build_prompt(
        question="Explain the model.",
        retrieved_chunks=[
            first_chunk,
            second_chunk,
        ],
    )

    assert "[Evidence 1]" in prompt
    assert "[Evidence 2]" in prompt
    assert "First retrieved evidence." in prompt
    assert "Second retrieved evidence." in prompt
    assert "Similarity Score: 0.9100" in prompt
    assert "Similarity Score: 0.8200" in prompt


def test_build_prompt_strips_question() -> None:
    builder = RagPromptBuilder()

    prompt = builder.build_prompt(
        question="   What is the Transformer?   ",
        retrieved_chunks=[build_search_result()],
    )

    assert "Question:\nWhat is the Transformer?" in prompt
    assert "   What is the Transformer?   " not in prompt


def test_build_prompt_rejects_empty_question() -> None:
    builder = RagPromptBuilder()

    with pytest.raises(RagPromptBuilderError, match="Question cannot be empty"):
        builder.build_prompt(
            question="   ",
            retrieved_chunks=[build_search_result()],
        )


def test_build_prompt_rejects_empty_retrieved_chunks() -> None:
    builder = RagPromptBuilder()

    with pytest.raises(
        RagPromptBuilderError,
        match="At least one retrieved chunk is required",
    ):
        builder.build_prompt(
            question="What is attention?",
            retrieved_chunks=[],
        )
