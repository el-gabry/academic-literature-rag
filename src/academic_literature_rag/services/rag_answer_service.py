from __future__ import annotations

from academic_literature_rag.models.rag_answer import (
    AnswerCitation,
    GroundedAnswer,
)
from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)
from academic_literature_rag.services.generation_client import (
    GenerationClient,
)
from academic_literature_rag.services.rag_prompt_builder import (
    RagPromptBuilder,
)
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchService,
)


class RagAnswerError(RuntimeError):
    """Raised when a grounded answer cannot be generated."""


class RagAnswerService:
    """Generates grounded answers from semantically retrieved chunks."""

    def __init__(
        self,
        *,
        semantic_search_service: SemanticSearchService,
        rag_prompt_builder: RagPromptBuilder,
        generation_client: GenerationClient,
    ) -> None:
        self._semantic_search_service = semantic_search_service
        self._rag_prompt_builder = rag_prompt_builder
        self._generation_client = generation_client

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
    ) -> GroundedAnswer:
        """Generate a grounded answer for a user question."""

        normalized_question = question.strip()

        if not normalized_question:
            raise RagAnswerError("Question cannot be empty.")

        retrieved_chunks = self._semantic_search_service.search(
            normalized_question,
            top_k=top_k,
        )

        if not retrieved_chunks:
            raise RagAnswerError("No relevant chunks were retrieved for the question.")

        prompt = self._rag_prompt_builder.build_prompt(
            question=normalized_question,
            retrieved_chunks=retrieved_chunks,
        )

        generation_response = self._generation_client.generate(prompt)

        if generation_response.model != self._generation_client.model_name:
            raise RagAnswerError("Generation response model does not match client model.")

        answer_text = generation_response.text.strip()

        if not answer_text:
            raise RagAnswerError("Generation client returned an empty answer.")

        return GroundedAnswer(
            question=normalized_question,
            answer=answer_text,
            citations=self._build_citations(retrieved_chunks),
            generation_model=generation_response.model,
        )

    @staticmethod
    def _build_citations(
        retrieved_chunks: list[SemanticSearchResult],
    ) -> list[AnswerCitation]:
        """Convert retrieved chunks into answer citations."""

        return [
            AnswerCitation(
                text_chunk_id=chunk.text_chunk_id,
                pdf_asset_id=chunk.pdf_asset_id,
                chunk_index=chunk.chunk_index,
                start_page_number=chunk.start_page_number,
                end_page_number=chunk.end_page_number,
                similarity_score=chunk.similarity_score,
                cited_text=chunk.text,
            )
            for chunk in retrieved_chunks
        ]
