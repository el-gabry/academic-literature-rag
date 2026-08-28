from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from academic_literature_rag.config import AppConfig
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


@dataclass(frozen=True)
class RagRepositories:
    """Repository bundle used by the application services."""

    search_run_repository: SearchRunRepository
    source_paper_repository: SourcePaperRepository
    canonical_paper_repository: CanonicalPaperRepository
    pdf_asset_repository: PdfAssetRepository
    pdf_page_text_repository: PdfPageTextRepository
    text_chunk_repository: TextChunkRepository
    chunk_embedding_repository: ChunkEmbeddingRepository


@dataclass(frozen=True)
class RagCoreServices:
    """Core service bundle used by pipeline and answer flows."""

    persisted_retrieval_service: PersistedRetrievalService
    pending_pdf_download_service: PendingPdfDownloadService
    pdf_text_extraction_service: PdfTextExtractionService
    text_chunking_service: TextChunkingService
    chunk_embedding_service: ChunkEmbeddingService
    semantic_search_service: SemanticSearchService
    rag_answer_service: RagAnswerService


class RagServiceFactory:
    """Builds configured repositories, clients, and RAG services."""

    def __init__(
        self,
        *,
        config: AppConfig,
        engine: Engine,
        session_factory: sessionmaker[Session],
        repositories: RagRepositories,
        arxiv_client: ArxivClient,
        pdf_download_service: PdfDownloadService,
    ) -> None:
        self._config = config
        self._engine = engine
        self._session_factory = session_factory
        self._repositories = repositories
        self._arxiv_client = arxiv_client
        self._pdf_download_service = pdf_download_service

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
    ) -> RagServiceFactory:
        """Create a factory from application configuration."""

        config.openai.require_api_key()
        config.storage.create_directories()

        engine = create_sqlite_engine(config.storage.database_path)
        create_schema(engine)
        session_factory = create_session_factory(engine)

        repositories = RagRepositories(
            search_run_repository=SearchRunRepository(session_factory),
            source_paper_repository=SourcePaperRepository(session_factory),
            canonical_paper_repository=CanonicalPaperRepository(session_factory),
            pdf_asset_repository=PdfAssetRepository(session_factory),
            pdf_page_text_repository=PdfPageTextRepository(session_factory),
            text_chunk_repository=TextChunkRepository(session_factory),
            chunk_embedding_repository=ChunkEmbeddingRepository(session_factory),
        )

        arxiv_client = ArxivClient()
        pdf_download_service = PdfDownloadService(
            pdf_asset_repository=repositories.pdf_asset_repository,
            pdf_storage_directory=config.storage.pdf_dir,
        )

        return cls(
            config=config,
            engine=engine,
            session_factory=session_factory,
            repositories=repositories,
            arxiv_client=arxiv_client,
            pdf_download_service=pdf_download_service,
        )

    @property
    def config(
        self,
    ) -> AppConfig:
        """Return application configuration."""

        return self._config

    @property
    def repositories(
        self,
    ) -> RagRepositories:
        """Return repository bundle."""

        return self._repositories

    @property
    def engine(
        self,
    ) -> Engine:
        """Return SQLAlchemy engine."""

        return self._engine

    @property
    def session_factory(
        self,
    ) -> sessionmaker[Session]:
        """Return SQLAlchemy session factory."""

        return self._session_factory

    def create_pipeline_service(
        self,
    ) -> RagPipelineService:
        """Create the ingestion pipeline service."""

        services = self.create_core_services()

        return RagPipelineService(
            persisted_retrieval_service=services.persisted_retrieval_service,
            pending_pdf_download_service=services.pending_pdf_download_service,
            pdf_text_extraction_service=services.pdf_text_extraction_service,
            text_chunking_service=services.text_chunking_service,
            chunk_embedding_service=services.chunk_embedding_service,
        )

    def create_answer_service(
        self,
    ) -> RagAnswerService:
        """Create the grounded answer service."""

        return self.create_core_services().rag_answer_service

    def create_core_services(
        self,
    ) -> RagCoreServices:
        """Create all core services needed by ingestion and answer flows."""

        persisted_retrieval_service = PersistedRetrievalService(
            client=self._arxiv_client,
            raw_response_store=RawResponseStore(self._config.storage.raw_response_dir),
            search_run_repository=self._repositories.search_run_repository,
            source_paper_repository=self._repositories.source_paper_repository,
            canonical_paper_repository=self._repositories.canonical_paper_repository,
            pdf_asset_repository=self._repositories.pdf_asset_repository,
        )

        pending_pdf_download_service = PendingPdfDownloadService(
            pdf_asset_repository=self._repositories.pdf_asset_repository,
            pdf_download_service=self._pdf_download_service,
        )

        pdf_text_extraction_service = PdfTextExtractionService(
            pdf_asset_repository=self._repositories.pdf_asset_repository,
            pdf_page_text_repository=self._repositories.pdf_page_text_repository,
        )

        text_chunking_service = TextChunkingService(
            pdf_page_text_repository=self._repositories.pdf_page_text_repository,
            text_chunk_repository=self._repositories.text_chunk_repository,
            text_cleaning_service=TextCleaningService(),
        )

        embedding_client = OpenAIEmbeddingClient(
            model_name=self._config.openai.embedding_model,
        )

        chunk_embedding_service = ChunkEmbeddingService(
            chunk_embedding_repository=self._repositories.chunk_embedding_repository,
            embedding_client=embedding_client,
        )

        semantic_search_service = SemanticSearchService(
            chunk_embedding_repository=self._repositories.chunk_embedding_repository,
            text_chunk_repository=self._repositories.text_chunk_repository,
            embedding_client=embedding_client,
        )

        rag_answer_service = RagAnswerService(
            semantic_search_service=semantic_search_service,
            rag_prompt_builder=RagPromptBuilder(),
            generation_client=OpenAIGenerationClient(
                model_name=self._config.openai.generation_model,
            ),
        )

        return RagCoreServices(
            persisted_retrieval_service=persisted_retrieval_service,
            pending_pdf_download_service=pending_pdf_download_service,
            pdf_text_extraction_service=pdf_text_extraction_service,
            text_chunking_service=text_chunking_service,
            chunk_embedding_service=chunk_embedding_service,
            semantic_search_service=semantic_search_service,
            rag_answer_service=rag_answer_service,
        )

    def close(
        self,
    ) -> None:
        """Close resources owned by the factory."""

        close_if_available(self._arxiv_client)
        close_if_available(self._pdf_download_service)
        self._engine.dispose()


def close_if_available(
    value: object,
) -> None:
    """Close a value if it exposes a close method."""

    close_method = getattr(
        value,
        "close",
        None,
    )

    if callable(close_method):
        close_method()