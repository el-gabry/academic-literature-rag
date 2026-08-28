from __future__ import annotations

from academic_literature_rag.app.factory import RagServiceFactory
from academic_literature_rag.config import AppConfig
from academic_literature_rag.models.rag_answer import GroundedAnswer
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)


def main() -> None:
    """Run one full RAG demo with real OpenAI embeddings and generation."""

    config = AppConfig.from_env()
    factory = RagServiceFactory.from_config(config)

    try:
        pipeline_service = factory.create_pipeline_service()

        ingestion_result = pipeline_service.ingest(
            query=config.demo.search_query,
            retrieval_limit=config.demo.retrieval_limit,
            download_limit=config.demo.download_limit,
            embedding_limit=config.demo.embedding_limit,
        )

        ensure_embeddings_exist(
            chunk_embedding_repository=factory.repositories.chunk_embedding_repository,
            embedding_model=config.openai.embedding_model,
        )

        answer_service = factory.create_answer_service()

        answer = answer_service.answer(
            config.demo.question,
            top_k=config.demo.top_k,
        )

    finally:
        factory.close()

    print_demo_result(
        search_query=config.demo.search_query,
        question=config.demo.question,
        embedding_model=config.openai.embedding_model,
        generation_model=config.openai.generation_model,
        ingestion_result=ingestion_result,
        answer=answer,
    )


def ensure_embeddings_exist(
    *,
    chunk_embedding_repository: ChunkEmbeddingRepository,
    embedding_model: str,
) -> None:
    """Fail clearly if no embeddings exist after ingestion."""

    embeddings = chunk_embedding_repository.list_by_model(embedding_model)

    if embeddings:
        return

    raise RuntimeError(
        f"No embeddings found for model '{embedding_model}' after ingestion. "
        "Try increasing RAG_DEMO_RETRIEVAL_LIMIT, RAG_DEMO_DOWNLOAD_LIMIT, "
        "or RAG_DEMO_EMBEDDING_LIMIT."
    )


def print_demo_result(
    *,
    search_query: str,
    question: str,
    embedding_model: str,
    generation_model: str,
    ingestion_result: object,
    answer: GroundedAnswer,
) -> None:
    """Print a readable full demo summary."""

    retrieval_result = ingestion_result.retrieval_result

    print("Academic Literature RAG end-to-end demo completed.")
    print()
    print("Configuration")
    print(f"- Search query: {search_query}")
    print(f"- Question: {question}")
    print(f"- Embedding model: {embedding_model}")
    print(f"- Generation model: {generation_model}")

    print()
    print("Retrieval")
    print(f"- Source: {retrieval_result.run.source}")
    print(f"- Run ID: {retrieval_result.run.run_id}")
    print(f"- Status: {retrieval_result.run.status}")
    print(f"- Papers persisted: {len(retrieval_result.papers)}")
    print(f"- Raw response: {retrieval_result.run.raw_response_path}")

    print()
    print("Papers")
    for index, paper in enumerate(
        retrieval_result.papers,
        start=1,
    ):
        print(f"{index}. {paper.title}")
        print(f"   Year: {paper.publication_year}")
        print(f"   PDF: {paper.open_access_pdf_url}")
        print(f"   Landing URL: {paper.landing_url}")

    print()
    print("PDF processing")
    if not ingestion_result.pdf_results:
        print("- No pending PDFs were downloaded in this run.")
    else:
        for pdf_result in ingestion_result.pdf_results:
            print(f"- PDF asset ID: {pdf_result.pdf_asset_id}")
            print(f"  URL: {pdf_result.source_url}")
            print(f"  Status: {pdf_result.status}")
            print(f"  Pages: {pdf_result.page_count}")
            print(f"  Chunks: {pdf_result.chunk_count}")
            print(f"  Error: {pdf_result.error_message}")

    print()
    print("Embeddings")
    if not ingestion_result.embedding_results:
        print("- No missing chunks were embedded in this run.")
    else:
        for embedding_result in ingestion_result.embedding_results:
            print(f"- Text chunk ID: {embedding_result.text_chunk_id}")
            print(f"  Status: {embedding_result.status}")
            print(f"  Model: {embedding_result.embedding_model}")
            print(f"  Error: {embedding_result.error_message}")

    print()
    print("Answer")
    print(answer.answer)

    print()
    print("Citations")
    for index, citation in enumerate(
        answer.citations,
        start=1,
    ):
        print(f"{index}. Chunk ID: {citation.text_chunk_id}")
        print(f"   PDF asset ID: {citation.pdf_asset_id}")
        print(f"   Chunk index: {citation.chunk_index}")
        print(f"   Pages: {citation.start_page_number}-{citation.end_page_number}")
        print(f"   Similarity: {citation.similarity_score:.4f}")
        print(f"   Text: {' '.join(citation.cited_text.split())[:500]}")


if __name__ == "__main__":
    main()