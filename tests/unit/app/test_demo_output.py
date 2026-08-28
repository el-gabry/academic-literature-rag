from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from academic_literature_rag.app.demo_output import (
    DemoOutputError,
    render_demo_output,
)
from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.models.rag_answer import AnswerCitation, GroundedAnswer
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.models.search_run import SearchRun
from academic_literature_rag.services.chunk_embedding_service import ChunkEmbeddingResult
from academic_literature_rag.services.rag_pipeline_service import (
    ProcessedPdfResult,
    RagPipelineIngestionResult,
)


def make_ingestion_result() -> RagPipelineIngestionResult:
    pdf_asset_id = uuid4()
    text_chunk_id = uuid4()

    return RagPipelineIngestionResult(
        retrieval_result=RetrievalResult(
            run=SearchRun(
                source="arxiv",
                query="transformer attention",
                status="completed",
                result_count=1,
                raw_response_path="data/raw_responses/arxiv/test.xml",
            ),
            papers=[
                PaperCandidate(
                    source="arxiv",
                    source_id="1706.03762",
                    title="Attention Is All You Need",
                    landing_url="https://arxiv.org/abs/1706.03762",
                    retrieved_at=datetime.now(UTC),
                    abstract="Transformer paper.",
                    authors=[
                        "Test Author",
                    ],
                    publication_year=2017,
                    venue="arXiv",
                    doi=None,
                    arxiv_id="1706.03762",
                    open_access_pdf_url="https://arxiv.org/pdf/1706.03762",
                    citation_count=None,
                )
            ],
        ),
        pdf_results=[
            ProcessedPdfResult(
                pdf_asset_id=pdf_asset_id,
                source_url="https://arxiv.org/pdf/1706.03762",
                status="processed",
                page_count=10,
                chunk_count=30,
            )
        ],
        embedding_results=[
            ChunkEmbeddingResult(
                text_chunk_id=text_chunk_id,
                status="embedded",
                embedding_model="text-embedding-3-small",
            )
        ],
    )


def make_answer() -> GroundedAnswer:
    text_chunk_id = uuid4()
    pdf_asset_id = uuid4()

    return GroundedAnswer(
        question="What is attention in transformers?",
        answer="Attention compares token representations and retrieves relevant context.",
        citations=[
            AnswerCitation(
                text_chunk_id=text_chunk_id,
                pdf_asset_id=pdf_asset_id,
                chunk_index=0,
                start_page_number=1,
                end_page_number=2,
                similarity_score=0.91,
                cited_text=(
                    "Attention mechanisms compare query and key vectors to weight "
                    "relevant token representations."
                ),
            )
        ],
        generation_model="gpt-4o-mini",
    )


def test_render_text_demo_output_contains_human_readable_sections() -> None:
    output = render_demo_output(
        search_query="transformer attention",
        question="What is attention in transformers?",
        embedding_model="text-embedding-3-small",
        generation_model="gpt-4o-mini",
        ingestion_result=make_ingestion_result(),
        answer=make_answer(),
        output_format="text",
    )

    assert "Academic Literature RAG end-to-end demo completed." in output
    assert "Configuration" in output
    assert "- Search query: transformer attention" in output
    assert "Retrieval" in output
    assert "Attention Is All You Need" in output
    assert "PDF processing" in output
    assert "Embeddings" in output
    assert "Answer" in output
    assert "Citations" in output
    assert "Similarity: 0.9100" in output


def test_render_json_demo_output_returns_machine_readable_payload() -> None:
    output = render_demo_output(
        search_query="transformer attention",
        question="What is attention in transformers?",
        embedding_model="text-embedding-3-small",
        generation_model="gpt-4o-mini",
        ingestion_result=make_ingestion_result(),
        answer=make_answer(),
        output_format="json",
    )

    payload = json.loads(output)

    assert payload["configuration"] == {
        "embedding_model": "text-embedding-3-small",
        "generation_model": "gpt-4o-mini",
        "question": "What is attention in transformers?",
        "search_query": "transformer attention",
    }
    assert payload["retrieval"]["source"] == "arxiv"
    assert payload["retrieval"]["status"] == "completed"
    assert payload["retrieval"]["papers_persisted"] == 1
    assert payload["papers"][0]["title"] == "Attention Is All You Need"
    assert payload["pdf_processing"][0]["status"] == "processed"
    assert payload["embeddings"][0]["status"] == "embedded"
    assert payload["answer"]["generation_model"] == "gpt-4o-mini"
    assert payload["citations"][0]["similarity_score"] == 0.91


def test_render_markdown_demo_output_contains_report_sections() -> None:
    output = render_demo_output(
        search_query="transformer attention",
        question="What is attention in transformers?",
        embedding_model="text-embedding-3-small",
        generation_model="gpt-4o-mini",
        ingestion_result=make_ingestion_result(),
        answer=make_answer(),
        output_format="markdown",
    )

    assert output.startswith("# Academic Literature RAG Demo Result")
    assert "## Configuration" in output
    assert "- **Search query:** transformer attention" in output
    assert "## Retrieval" in output
    assert "## Papers" in output
    assert "1. **Attention Is All You Need**" in output
    assert "## PDF Processing" in output
    assert "## Embeddings" in output
    assert "## Answer" in output
    assert "## Citations" in output
    assert "### Citation 1" in output


def test_render_demo_output_rejects_unsupported_format() -> None:
    with pytest.raises(
        DemoOutputError,
        match="Unsupported demo output format",
    ):
        render_demo_output(
            search_query="transformer attention",
            question="What is attention in transformers?",
            embedding_model="text-embedding-3-small",
            generation_model="gpt-4o-mini",
            ingestion_result=make_ingestion_result(),
            answer=make_answer(),
            output_format="xml",  # type: ignore[arg-type]
        )


def test_text_output_handles_empty_pdf_and_embedding_results() -> None:
    ingestion_result = RagPipelineIngestionResult(
        retrieval_result=RetrievalResult(
            run=SearchRun(
                source="arxiv",
                query="transformer attention",
                status="completed",
                result_count=0,
            ),
            papers=[],
        ),
        pdf_results=[],
        embedding_results=[],
    )

    output = render_demo_output(
        search_query="transformer attention",
        question="What is attention in transformers?",
        embedding_model="text-embedding-3-small",
        generation_model="gpt-4o-mini",
        ingestion_result=ingestion_result,
        answer=make_answer(),
        output_format="text",
    )

    assert "- No papers persisted." in output
    assert "- No pending PDFs were downloaded in this run." in output
    assert "- No missing chunks were embedded in this run." in output