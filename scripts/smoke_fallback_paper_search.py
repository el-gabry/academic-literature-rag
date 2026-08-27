from __future__ import annotations

import os

from academic_literature_rag.connectors.arxiv import ArxivClient
from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
)
from academic_literature_rag.services.fallback_paper_search_service import (
    FallbackPaperSearchError,
    FallbackPaperSearchService,
)


def main() -> None:
    query = "retrieval augmented generation scientific literature"
    limit = 5

    semantic_scholar_client = SemanticScholarClient(
        api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )
    arxiv_client = ArxivClient()

    service = FallbackPaperSearchService(
        primary_client=semantic_scholar_client,
        fallback_client=arxiv_client,
    )

    try:
        result = service.search(
            query=query,
            limit=limit,
        )
    except FallbackPaperSearchError as error:
        print("Search failed.")
        print(error)
        raise SystemExit(1) from error
    finally:
        semantic_scholar_client.close()
        arxiv_client.close()

    print("Search succeeded.")
    print(f"Query: {result.query}")
    print(f"Selected source: {result.source_name}")
    print(f"Fallback used: {result.fallback_used}")
    print(f"Papers found: {len(result.papers)}")

    if result.failures:
        print()
        print("Failures before success:")
        for failure in result.failures:
            print(
                f"- {failure.source_name}: "
                f"{failure.error_type}: {failure.message}"
            )

    print()
    print("Papers:")
    for index, paper in enumerate(result.papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   Source: {paper.source}")
        print(f"   Source ID: {paper.source_id}")
        print(f"   Year: {paper.publication_year}")
        print(f"   PDF: {paper.open_access_pdf_url}")
        print(f"   Landing URL: {paper.landing_url}")
        print()


if __name__ == "__main__":
    main()