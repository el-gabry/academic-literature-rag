from __future__ import annotations

from academic_literature_rag.models.semantic_search_result import (
    SemanticSearchResult,
)


class RagPromptBuilderError(ValueError):
    """Raised when a RAG prompt cannot be built."""


class RagPromptBuilder:
    """Builds grounded-answer prompts from retrieved text chunks."""

    def build_prompt(
        self,
        *,
        question: str,
        retrieved_chunks: list[SemanticSearchResult],
    ) -> str:
        """Build a prompt that asks the model to answer using evidence only."""

        normalized_question = question.strip()

        if not normalized_question:
            raise RagPromptBuilderError("Question cannot be empty.")

        if not retrieved_chunks:
            raise RagPromptBuilderError("At least one retrieved chunk is required.")

        evidence_block = self._build_evidence_block(retrieved_chunks)

        return (
            "You are an academic research assistant.\n"
            "Answer the question using only the provided evidence.\n"
            "Do not use outside knowledge.\n"
            "If the evidence is insufficient, say that the evidence is insufficient.\n"
            "Keep the answer concise and grounded.\n\n"
            f"Question:\n{normalized_question}\n\n"
            f"Evidence:\n{evidence_block}\n\n"
            "Answer:"
        )

    @staticmethod
    def _build_evidence_block(
        retrieved_chunks: list[SemanticSearchResult],
    ) -> str:
        """Format retrieved chunks as numbered evidence blocks."""

        evidence_items = []

        for index, chunk in enumerate(retrieved_chunks, start=1):
            evidence_items.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"Chunk ID: {chunk.text_chunk_id}",
                        f"PDF Asset ID: {chunk.pdf_asset_id}",
                        f"Chunk Index: {chunk.chunk_index}",
                        (f"Pages: {chunk.start_page_number}-{chunk.end_page_number}"),
                        f"Similarity Score: {chunk.similarity_score:.4f}",
                        "Text:",
                        chunk.text,
                    ]
                )
            )

        return "\n\n".join(evidence_items)
