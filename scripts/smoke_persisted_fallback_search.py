from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.orm import sessionmaker

from academic_literature_rag.connectors.arxiv import ArxivClient
from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
)
from academic_literature_rag.database.session import (
    create_schema,
    create_sqlite_engine,
)
from academic_literature_rag.models.retrieval_result import RetrievalResult
from academic_literature_rag.repositories.canonical_paper_repository import (
    CanonicalPaperRepository,
)
from academic_literature_rag.repositories.pdf_asset_repository import (
    PdfAssetRepository,
)
from academic_literature_rag.repositories.search_run_repository import (
    SearchRunRepository,
)
from academic_literature_rag.repositories.source_paper_repository import (
    SourcePaperRepository,
)
from academic_literature_rag.services.persisted_retrieval_service import (
    PersistedRetrievalService,
)
from academic_literature_rag.storage.raw_response_store import RawResponseStore


def main() -> None:
    query = "retrieval augmented generation scientific literature"
    limit = 5

    database_path = Path("data/db/dev.db")
    raw_response_dir = Path("data/raw_responses")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)

    engine = create_sqlite_engine(database_path)
    create_schema(engine)

    session_factory = sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    search_run_repository = SearchRunRepository(session_factory)
    source_paper_repository = SourcePaperRepository(session_factory)
    canonical_paper_repository = CanonicalPaperRepository(session_factory)
    pdf_asset_repository = PdfAssetRepository(session_factory)
    raw_response_store = RawResponseStore(raw_response_dir)

    semantic_scholar_client = SemanticScholarClient(
        api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )
    arxiv_client = ArxivClient()

    try:
        result = search_with_persisted_fallback(
            query=query,
            limit=limit,
            semantic_scholar_client=semantic_scholar_client,
            arxiv_client=arxiv_client,
            raw_response_store=raw_response_store,
            search_run_repository=search_run_repository,
            source_paper_repository=source_paper_repository,
            canonical_paper_repository=canonical_paper_repository,
            pdf_asset_repository=pdf_asset_repository,
        )
    finally:
        semantic_scholar_client.close()
        arxiv_client.close()

    pending_pdf_assets = pdf_asset_repository.list_pending(limit=20)

    print("Persisted search succeeded.")
    print(f"Selected source: {result.run.source}")
    print(f"Run ID: {result.run.run_id}")
    print(f"Status: {result.run.status}")
    print(f"Raw response: {result.run.raw_response_path}")
    print(f"Papers persisted: {len(result.papers)}")
    print(f"Pending PDF assets: {len(pending_pdf_assets)}")

    print()
    print("Papers:")
    for index, paper in enumerate(result.papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   Source: {paper.source}")
        print(f"   Source ID: {paper.source_id}")
        print(f"   Year: {paper.publication_year}")
        print(f"   PDF candidate: {paper.open_access_pdf_url}")
        print(f"   Landing URL: {paper.landing_url}")
        print()

    print("Pending PDFs:")
    for index, pdf_asset in enumerate(pending_pdf_assets, start=1):
        print(f"{index}. {pdf_asset.source_url}")
        print(f"   PDF asset ID: {pdf_asset.pdf_asset_id}")
        print(f"   Canonical paper ID: {pdf_asset.canonical_paper_id}")
        print(f"   Status: {pdf_asset.download_status}")
        print()


def search_with_persisted_fallback(
    *,
    query: str,
    limit: int,
    semantic_scholar_client: SemanticScholarClient,
    arxiv_client: ArxivClient,
    raw_response_store: RawResponseStore,
    search_run_repository: SearchRunRepository,
    source_paper_repository: SourcePaperRepository,
    canonical_paper_repository: CanonicalPaperRepository,
    pdf_asset_repository: PdfAssetRepository,
) -> RetrievalResult:
    semantic_scholar_service = PersistedRetrievalService(
        client=semantic_scholar_client,
        raw_response_store=raw_response_store,
        search_run_repository=search_run_repository,
        source_paper_repository=source_paper_repository,
        canonical_paper_repository=canonical_paper_repository,
        pdf_asset_repository=pdf_asset_repository,
    )

    arxiv_service = PersistedRetrievalService(
        client=arxiv_client,
        raw_response_store=raw_response_store,
        search_run_repository=search_run_repository,
        source_paper_repository=source_paper_repository,
        canonical_paper_repository=canonical_paper_repository,
        pdf_asset_repository=pdf_asset_repository,
    )

    try:
        semantic_scholar_result = semantic_scholar_service.search(
            query=query,
            limit=limit,
        )

        if semantic_scholar_result.papers:
            return semantic_scholar_result

        print("Semantic Scholar returned no papers. Falling back to arXiv.")

    except Exception as error:
        print("Semantic Scholar persisted search failed.")
        print(f"{type(error).__name__}: {error}")
        print("Falling back to arXiv.")

    arxiv_result = arxiv_service.search(
        query=query,
        limit=limit,
    )

    if not arxiv_result.papers:
        raise RuntimeError("Both Semantic Scholar and arXiv returned no papers.")

    return arxiv_result


if __name__ == "__main__":
    main()