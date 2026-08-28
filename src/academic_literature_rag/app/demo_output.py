from __future__ import annotations

import json
from typing import Any, Literal

from academic_literature_rag.models.rag_answer import GroundedAnswer
from academic_literature_rag.services.rag_pipeline_service import (
    RagPipelineIngestionResult,
)

DemoOutputFormat = Literal[
    "text",
    "json",
    "markdown",
]

SUPPORTED_DEMO_OUTPUT_FORMATS: tuple[str, ...] = (
    "text",
    "json",
    "markdown",
)


class DemoOutputError(ValueError):
    """Raised when demo output cannot be rendered."""


def render_demo_output(
    *,
    search_query: str,
    question: str,
    embedding_model: str,
    generation_model: str,
    ingestion_result: RagPipelineIngestionResult,
    answer: GroundedAnswer,
    output_format: DemoOutputFormat = "text",
) -> str:
    """Render a full demo result in the requested output format."""

    if output_format == "text":
        return render_text_demo_output(
            search_query=search_query,
            question=question,
            embedding_model=embedding_model,
            generation_model=generation_model,
            ingestion_result=ingestion_result,
            answer=answer,
        )

    if output_format == "json":
        return render_json_demo_output(
            search_query=search_query,
            question=question,
            embedding_model=embedding_model,
            generation_model=generation_model,
            ingestion_result=ingestion_result,
            answer=answer,
        )

    if output_format == "markdown":
        return render_markdown_demo_output(
            search_query=search_query,
            question=question,
            embedding_model=embedding_model,
            generation_model=generation_model,
            ingestion_result=ingestion_result,
            answer=answer,
        )

    raise DemoOutputError(f"Unsupported demo output format: {output_format}")


def render_text_demo_output(
    *,
    search_query: str,
    question: str,
    embedding_model: str,
    generation_model: str,
    ingestion_result: RagPipelineIngestionResult,
    answer: GroundedAnswer,
) -> str:
    """Render a full demo result as terminal-friendly text."""

    retrieval_result = ingestion_result.retrieval_result
    lines: list[str] = []

    lines.extend(
        [
            "Academic Literature RAG end-to-end demo completed.",
            "",
            "Configuration",
            f"- Search query: {search_query}",
            f"- Question: {question}",
            f"- Embedding model: {embedding_model}",
            f"- Generation model: {generation_model}",
            "",
            "Retrieval",
            f"- Source: {retrieval_result.run.source}",
            f"- Run ID: {retrieval_result.run.run_id}",
            f"- Status: {retrieval_result.run.status}",
            f"- Papers persisted: {len(retrieval_result.papers)}",
            f"- Raw response: {retrieval_result.run.raw_response_path}",
            "",
            "Papers",
        ]
    )

    if retrieval_result.papers:
        for index, paper in enumerate(
            retrieval_result.papers,
            start=1,
        ):
            lines.extend(
                [
                    f"{index}. {paper.title}",
                    f"   Year: {paper.publication_year}",
                    f"   PDF: {paper.open_access_pdf_url}",
                    f"   Landing URL: {paper.landing_url}",
                ]
            )
    else:
        lines.append("- No papers persisted.")

    lines.extend(
        [
            "",
            "PDF processing",
        ]
    )

    if ingestion_result.pdf_results:
        for pdf_result in ingestion_result.pdf_results:
            lines.extend(
                [
                    f"- PDF asset ID: {pdf_result.pdf_asset_id}",
                    f"  URL: {pdf_result.source_url}",
                    f"  Status: {pdf_result.status}",
                    f"  Pages: {pdf_result.page_count}",
                    f"  Chunks: {pdf_result.chunk_count}",
                    f"  Error: {pdf_result.error_message}",
                ]
            )
    else:
        lines.append("- No pending PDFs were downloaded in this run.")

    lines.extend(
        [
            "",
            "Embeddings",
        ]
    )

    if ingestion_result.embedding_results:
        for embedding_result in ingestion_result.embedding_results:
            lines.extend(
                [
                    f"- Text chunk ID: {embedding_result.text_chunk_id}",
                    f"  Status: {embedding_result.status}",
                    f"  Model: {embedding_result.embedding_model}",
                    f"  Error: {embedding_result.error_message}",
                ]
            )
    else:
        lines.append("- No missing chunks were embedded in this run.")

    lines.extend(
        [
            "",
            "Answer",
            answer.answer,
            "",
            "Citations",
        ]
    )

    for index, citation in enumerate(
        answer.citations,
        start=1,
    ):
        lines.extend(
            [
                f"{index}. Chunk ID: {citation.text_chunk_id}",
                f"   PDF asset ID: {citation.pdf_asset_id}",
                f"   Chunk index: {citation.chunk_index}",
                f"   Pages: {citation.start_page_number}-{citation.end_page_number}",
                f"   Similarity: {citation.similarity_score:.4f}",
                f"   Text: {_compact_text(citation.cited_text, max_length=500)}",
            ]
        )

    return "\n".join(lines)


def render_json_demo_output(
    *,
    search_query: str,
    question: str,
    embedding_model: str,
    generation_model: str,
    ingestion_result: RagPipelineIngestionResult,
    answer: GroundedAnswer,
) -> str:
    """Render a full demo result as JSON."""

    payload = build_demo_output_payload(
        search_query=search_query,
        question=question,
        embedding_model=embedding_model,
        generation_model=generation_model,
        ingestion_result=ingestion_result,
        answer=answer,
    )

    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
    )


def render_markdown_demo_output(
    *,
    search_query: str,
    question: str,
    embedding_model: str,
    generation_model: str,
    ingestion_result: RagPipelineIngestionResult,
    answer: GroundedAnswer,
) -> str:
    """Render a full demo result as Markdown."""

    retrieval_result = ingestion_result.retrieval_result
    lines: list[str] = []

    lines.extend(
        [
            "# Academic Literature RAG Demo Result",
            "",
            "## Configuration",
            "",
            f"- **Search query:** {search_query}",
            f"- **Question:** {question}",
            f"- **Embedding model:** `{embedding_model}`",
            f"- **Generation model:** `{generation_model}`",
            "",
            "## Retrieval",
            "",
            f"- **Source:** {retrieval_result.run.source}",
            f"- **Run ID:** `{retrieval_result.run.run_id}`",
            f"- **Status:** {retrieval_result.run.status}",
            f"- **Papers persisted:** {len(retrieval_result.papers)}",
            f"- **Raw response:** `{retrieval_result.run.raw_response_path}`",
            "",
            "## Papers",
            "",
        ]
    )

    if retrieval_result.papers:
        for index, paper in enumerate(
            retrieval_result.papers,
            start=1,
        ):
            lines.extend(
                [
                    f"{index}. **{paper.title}**",
                    f"   - Year: {paper.publication_year}",
                    f"   - PDF: {paper.open_access_pdf_url}",
                    f"   - Landing URL: {paper.landing_url}",
                ]
            )
    else:
        lines.append("- No papers persisted.")

    lines.extend(
        [
            "",
            "## PDF Processing",
            "",
        ]
    )

    if ingestion_result.pdf_results:
        for pdf_result in ingestion_result.pdf_results:
            lines.extend(
                [
                    f"- **PDF asset ID:** `{pdf_result.pdf_asset_id}`",
                    f"  - URL: {pdf_result.source_url}",
                    f"  - Status: {pdf_result.status}",
                    f"  - Pages: {pdf_result.page_count}",
                    f"  - Chunks: {pdf_result.chunk_count}",
                    f"  - Error: {pdf_result.error_message}",
                ]
            )
    else:
        lines.append("- No pending PDFs were downloaded in this run.")

    lines.extend(
        [
            "",
            "## Embeddings",
            "",
        ]
    )

    if ingestion_result.embedding_results:
        for embedding_result in ingestion_result.embedding_results:
            lines.extend(
                [
                    f"- **Text chunk ID:** `{embedding_result.text_chunk_id}`",
                    f"  - Status: {embedding_result.status}",
                    f"  - Model: `{embedding_result.embedding_model}`",
                    f"  - Error: {embedding_result.error_message}",
                ]
            )
    else:
        lines.append("- No missing chunks were embedded in this run.")

    lines.extend(
        [
            "",
            "## Answer",
            "",
            answer.answer,
            "",
            "## Citations",
            "",
        ]
    )

    for index, citation in enumerate(
        answer.citations,
        start=1,
    ):
        lines.extend(
            [
                f"### Citation {index}",
                "",
                f"- **Chunk ID:** `{citation.text_chunk_id}`",
                f"- **PDF asset ID:** `{citation.pdf_asset_id}`",
                f"- **Chunk index:** {citation.chunk_index}",
                f"- **Pages:** {citation.start_page_number}-{citation.end_page_number}",
                f"- **Similarity:** {citation.similarity_score:.4f}",
                "",
                "> " + _compact_text(citation.cited_text, max_length=700),
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def build_demo_output_payload(
    *,
    search_query: str,
    question: str,
    embedding_model: str,
    generation_model: str,
    ingestion_result: RagPipelineIngestionResult,
    answer: GroundedAnswer,
) -> dict[str, Any]:
    """Build a JSON-serializable demo output payload."""

    retrieval_result = ingestion_result.retrieval_result

    return {
        "configuration": {
            "search_query": search_query,
            "question": question,
            "embedding_model": embedding_model,
            "generation_model": generation_model,
        },
        "retrieval": {
            "source": retrieval_result.run.source,
            "run_id": str(retrieval_result.run.run_id),
            "status": retrieval_result.run.status,
            "papers_persisted": len(retrieval_result.papers),
            "raw_response_path": retrieval_result.run.raw_response_path,
        },
        "papers": [
            {
                "source": paper.source,
                "source_id": paper.source_id,
                "title": paper.title,
                "publication_year": paper.publication_year,
                "venue": paper.venue,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "citation_count": paper.citation_count,
                "open_access_pdf_url": paper.open_access_pdf_url,
                "landing_url": paper.landing_url,
            }
            for paper in retrieval_result.papers
        ],
        "pdf_processing": [
            {
                "pdf_asset_id": str(pdf_result.pdf_asset_id),
                "source_url": pdf_result.source_url,
                "status": pdf_result.status,
                "page_count": pdf_result.page_count,
                "chunk_count": pdf_result.chunk_count,
                "error_message": pdf_result.error_message,
            }
            for pdf_result in ingestion_result.pdf_results
        ],
        "embeddings": [
            {
                "text_chunk_id": str(embedding_result.text_chunk_id),
                "status": embedding_result.status,
                "embedding_model": embedding_result.embedding_model,
                "error_message": embedding_result.error_message,
            }
            for embedding_result in ingestion_result.embedding_results
        ],
        "answer": {
            "question": answer.question,
            "answer": answer.answer,
            "generation_model": answer.generation_model,
            "created_at": answer.created_at.isoformat(),
        },
        "citations": [
            {
                "text_chunk_id": str(citation.text_chunk_id),
                "pdf_asset_id": str(citation.pdf_asset_id),
                "chunk_index": citation.chunk_index,
                "start_page_number": citation.start_page_number,
                "end_page_number": citation.end_page_number,
                "similarity_score": citation.similarity_score,
                "cited_text": citation.cited_text,
            }
            for citation in answer.citations
        ],
    }


def _compact_text(
    value: str,
    *,
    max_length: int,
) -> str:
    """Compact text whitespace and truncate to a maximum length."""

    compacted = " ".join(value.split())

    if len(compacted) <= max_length:
        return compacted

    return compacted[: max_length - 3] + "..."