from __future__ import annotations

from pathlib import Path

from academic_literature_rag.connectors.arxiv import ArxivClient
from academic_literature_rag.database.session import (
    create_schema,
    create_session_factory,
    create_sqlite_engine,
)
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
from academic_literature_rag.services.embedding_client import EmbeddingResponse
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
from academic_literature_rag.services.rag_pipeline_service import (
    RagPipelineService,
)
from academic_literature_rag.services.text_chunking_service import (
    TextChunkingService,
)
from academic_literature_rag.services.text_cleaning_service import (
    TextCleaningService,
)
from academic_literature_rag.storage.raw_response_store import RawResponseStore


class SmokeEmbeddingClient:
    """Small deterministic embedding client for local smoke runs."""

    model_name = "smoke-deterministic-embedding"

    def embed_text(
        self,
        text: str,
    ) -> EmbeddingResponse:
        """Return a deterministic vector without calling an external API."""

        vector = self._build_vector(text)

        try:
            return EmbeddingResponse(
                vector=vector,
                model=self.model_name,
                dimension=len(vector),
            )
        except TypeError:
            return EmbeddingResponse(
                vector=vector,
                model=self.model_name,
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
    """Run one local RAG ingestion smoke test."""

    query = "transformer attention"
    retrieval_limit = 2
    download_limit = 1
    embedding_limit = 3

    database_path = Path("data/db/dev.db")
    raw_response_dir = Path("data/raw_responses")
    pdf_dir = Path("data/pdfs")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

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
            embedding_client=SmokeEmbeddingClient(),
        )

        pipeline_service = RagPipelineService(
            persisted_retrieval_service=persisted_retrieval_service,
            pending_pdf_download_service=pending_pdf_download_service,
            pdf_text_extraction_service=pdf_text_extraction_service,
            text_chunking_service=text_chunking_service,
            chunk_embedding_service=chunk_embedding_service,
        )

        result = pipeline_service.ingest(
            query=query,
            retrieval_limit=retrieval_limit,
            download_limit=download_limit,
            embedding_limit=embedding_limit,
        )

    finally:
        close_if_available(arxiv_client)
        close_if_available(pdf_download_service)

    print_result(result)





def close_if_available(
    value: object,
) -> None:
    """Close clients/services that expose a close method."""

    close_method = getattr(value, "close", None)

    if callable(close_method):
        close_method()


def print_result(
    result: object,
) -> None:
    """Print a readable summary of one pipeline run."""

    retrieval_result = result.retrieval_result

    print("RAG pipeline smoke run completed.")
    print()
    print("Retrieval")
    print(f"- Source: {retrieval_result.run.source}")
    print(f"- Run ID: {retrieval_result.run.run_id}")
    print(f"- Status: {retrieval_result.run.status}")
    print(f"- Papers persisted: {len(retrieval_result.papers)}")
    print(f"- Raw response: {retrieval_result.run.raw_response_path}")

    print()
    print("Papers")

    for index, paper in enumerate(retrieval_result.papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   Year: {paper.publication_year}")
        print(f"   PDF: {paper.open_access_pdf_url}")
        print(f"   Landing URL: {paper.landing_url}")

    print()
    print("PDF processing")

    if not result.pdf_results:
        print("- No pending PDFs were downloaded.")
    else:
        for pdf_result in result.pdf_results:
            print(f"- PDF asset ID: {pdf_result.pdf_asset_id}")
            print(f"  URL: {pdf_result.source_url}")
            print(f"  Status: {pdf_result.status}")
            print(f"  Pages: {pdf_result.page_count}")
            print(f"  Chunks: {pdf_result.chunk_count}")
            print(f"  Error: {pdf_result.error_message}")

    print()
    print("Embeddings")

    if not result.embedding_results:
        print("- No chunks were embedded.")
    else:
        for embedding_result in result.embedding_results:
            print(f"- Text chunk ID: {embedding_result.text_chunk_id}")
            print(f"  Status: {embedding_result.status}")
            print(f"  Model: {embedding_result.embedding_model}")
            print(f"  Error: {embedding_result.error_message}")


if __name__ == "__main__":
    main()