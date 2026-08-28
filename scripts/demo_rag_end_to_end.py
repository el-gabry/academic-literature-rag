from __future__ import annotations

import os
from pathlib import Path

from academic_literature_rag.connectors.arxiv import ArxivClient
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
from academic_literature_rag.models.rag_answer import GroundedAnswer
from academic_literature_rag.repositories.canonical_paper_repository import (
    CanonicalPaperRepository,
)
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.repositories.pdf_page_text_repository import (
    PdfPageTextRepository,
)
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)
from academic_literature_rag.repositories.source_paper_repository import (
    SourcePaperRepository,
)
from academic_literature_rag.repositories.text_chunk_repository import (
    TextChunkRepository,
)
from academic_literature_rag.services.chunk_embedding_service import (
    ChunkEmbeddingService,
)
from academic_literature_rag.services.openai_embedding_client import (
    OpenAIEmbeddingClient,
)
from academic_literature_rag.services.openai_generation_client import (
    OpenAIGenerationClient,
)
from academic_literature_rag.services.pdf_download_service import (
    PdfDownloadService,
)
from academic_literature_rag.services.pdf_text_extraction_service import (
    PdfTextExtractionService,
)
from academic_literature_rag.services.pending_pdf_download_service import (
    PendingPdfDownloadService,
)
from academic_literature_rag.services.persisted_retrieval_service import (
    PersistedRetrievalService,
)
from academic_literature_rag.services.rag_answer_service import (
    RagAnswerService,
)
from academic_literature_rag.services.rag_pipeline_service import (
    RagPipelineService,
)
from academic_literature_rag.services.rag_prompt_builder import (
    RagPromptBuilder,
)
from academic_literature_rag.services.semantic_search_service import (
    SemanticSearchService,
)
from academic_literature_rag.services.text_chunking_service import (
    TextChunkingService,
)
from academic_literature_rag.services.text_cleaning_service import (
    TextCleaningService,
)
from academic_literature_rag.storage.raw_response_store import RawResponseStore


def main() -> None:
    """Run one full RAG demo with real OpenAI embeddings and generation."""

    ensure_openai_api_key_exists()

    search_query = os.getenv(
        "RAG_DEMO_SEARCH_QUERY",
        "transformer attention",
    )
    question = os.getenv(
        "RAG_DEMO_QUESTION",
        "What is attention in transformers?",
    )

    retrieval_limit = int(
        os.getenv(
            "RAG_DEMO_RETRIEVAL_LIMIT",
            "2",
        )
    )
    download_limit = int(
        os.getenv(
            "RAG_DEMO_DOWNLOAD_LIMIT",
            "1",
        )
    )
    embedding_limit = int(
        os.getenv(
            "RAG_DEMO_EMBEDDING_LIMIT",
            "8",
        )
    )
    top_k = int(
        os.getenv(
            "RAG_DEMO_TOP_K",
            "3",
        )
    )

    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    generation_model = os.getenv(
        "OPENAI_GENERATION_MODEL",
        "gpt-4o-mini",
    )

    database_path = Path(
        os.getenv(
            "RAG_DEMO_DATABASE_PATH",
            "data/db/dev.db",
        )
    )
    raw_response_dir = Path(
        os.getenv(
            "RAG_DEMO_RAW_RESPONSE_DIR",
            "data/raw_responses",
        )
    )
    pdf_dir = Path(
        os.getenv(
            "RAG_DEMO_PDF_DIR",
            "data/pdfs",
        )
    )

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    raw_response_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    pdf_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    engine = create_sqlite_engine(database_path)
    create_schema(engine)
    session_factory = create_session_factory(engine)

    search_run_repository = SearchRunRepository(session_factory)
    source_paper_repository = SourcePaperRepository(session_factory)
    canonical_paper_repository = CanonicalPaperRepository(session_factory)
    pdf_asset_repository = PdfAssetRepository(session_factory)
    pdf_page_text_repository = PdfPageTextRepository(session_factory)
    text_chunk_repository = TextChunkRepository(session_factory)
    chunk_embedding_repository = ChunkEmbeddingRepository(session_factory)

    embedding_client = OpenAIEmbeddingClient(
        model_name=embedding_model,
    )

    arxiv_client = ArxivClient()
    pdf_download_service = PdfDownloadService(
        pdf_asset_repository=pdf_asset_repository,
        pdf_storage_directory=pdf_dir,
    )

    try:
        persisted_retrieval_service = PersistedRetrievalService(
            client=arxiv_client,
            raw_response_store=RawResponseStore(raw_response_dir),
            search_run_repository=search_run_repository,
            source_paper_repository=source_paper_repository,
            canonical_paper_repository=canonical_paper_repository,
            pdf_asset_repository=pdf_asset_repository,
        )

        pending_pdf_download_service = PendingPdfDownloadService(
            pdf_asset_repository=pdf_asset_repository,
            pdf_download_service=pdf_download_service,
        )

        pdf_text_extraction_service = PdfTextExtractionService(
            pdf_asset_repository=pdf_asset_repository,
            pdf_page_text_repository=pdf_page_text_repository,
        )

        text_chunking_service = TextChunkingService(
            pdf_page_text_repository=pdf_page_text_repository,
            text_chunk_repository=text_chunk_repository,
            text_cleaning_service=TextCleaningService(),
        )

        chunk_embedding_service = ChunkEmbeddingService(
            chunk_embedding_repository=chunk_embedding_repository,
            embedding_client=embedding_client,
        )

        pipeline_service = RagPipelineService(
            persisted_retrieval_service=persisted_retrieval_service,
            pending_pdf_download_service=pending_pdf_download_service,
            pdf_text_extraction_service=pdf_text_extraction_service,
            text_chunking_service=text_chunking_service,
            chunk_embedding_service=chunk_embedding_service,
        )

        ingestion_result = pipeline_service.ingest(
            query=search_query,
            retrieval_limit=retrieval_limit,
            download_limit=download_limit,
            embedding_limit=embedding_limit,
        )

        ensure_embeddings_exist(
            chunk_embedding_repository=chunk_embedding_repository,
            embedding_model=embedding_model,
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
                model_name=generation_model,
            ),
        )

        answer = rag_answer_service.answer(
            question,
            top_k=top_k,
        )

    finally:
        close_if_available(arxiv_client)
        close_if_available(pdf_download_service)

    print_demo_result(
        search_query=search_query,
        question=question,
        embedding_model=embedding_model,
        generation_model=generation_model,
        ingestion_result=ingestion_result,
        answer=answer,
    )


def ensure_openai_api_key_exists() -> None:
    """Fail early if OPENAI_API_KEY is missing."""

    if os.getenv("OPENAI_API_KEY"):
        return

    raise RuntimeError(
        "OPENAI_API_KEY is not set. Set it before running this script:\n"
        "export OPENAI_API_KEY='your-key-here'"
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


def close_if_available(
    value: object,
) -> None:
    """Close clients/services that expose a close method."""

    close_method = getattr(
        value,
        "close",
        None,
    )

    if callable(close_method):
        close_method()


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