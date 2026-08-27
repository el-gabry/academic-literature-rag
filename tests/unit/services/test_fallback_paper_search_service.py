from __future__ import annotations

from datetime import UTC, datetime
import pytest

from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.services.fallback_paper_search_service import (
    FallbackPaperSearchError,
    FallbackPaperSearchService,
)


class FakePaperSearchClient:
    """Fake paper-search client for fallback-search tests."""

    def __init__(
        self,
        *,
        source_name: str,
        papers: list[PaperCandidate] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.source_name = source_name
        self._papers = papers or []
        self._error = error
        self.search_calls: list[dict[str, object]] = []

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperCandidate]:
        self.search_calls.append(
            {
                "query": query,
                "limit": limit,
            }
        )

        if self._error is not None:
            raise self._error

        return self._papers


def make_paper_candidate() -> PaperCandidate:
    """Create a lightweight paper candidate for service tests."""

    return PaperCandidate(
        source="semantic_scholar",
        source_id="test-paper-id",
        title="Attention Is All You Need",
        landing_url="https://example.com/paper",
        retrieved_at=datetime.now(UTC),
        abstract="A test abstract.",
        authors=["Test Author"],
        publication_year=2017,
        venue="Test Venue",
        doi=None,
        arxiv_id="1706.03762",
        open_access_pdf_url="https://example.com/paper.pdf",
        citation_count=100,
    )


def test_search_uses_primary_when_primary_returns_papers() -> None:
    paper = make_paper_candidate()

    primary_client = FakePaperSearchClient(
        source_name="semantic_scholar",
        papers=[paper],
    )
    fallback_client = FakePaperSearchClient(
        source_name="arxiv",
        papers=[make_paper_candidate()],
    )

    service = FallbackPaperSearchService(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )

    result = service.search(
        query="  attention is all you need  ",
        limit=3,
    )

    assert result.source_name == "semantic_scholar"
    assert result.query == "attention is all you need"
    assert result.papers == [paper]
    assert result.fallback_used is False
    assert result.failures == ()

    assert primary_client.search_calls == [
        {
            "query": "attention is all you need",
            "limit": 3,
        }
    ]
    assert fallback_client.search_calls == []


def test_search_falls_back_when_primary_raises_error() -> None:
    fallback_paper = make_paper_candidate()

    primary_client = FakePaperSearchClient(
        source_name="semantic_scholar",
        error=RuntimeError("quota exceeded"),
    )
    fallback_client = FakePaperSearchClient(
        source_name="arxiv",
        papers=[fallback_paper],
    )

    service = FallbackPaperSearchService(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )

    result = service.search(
        query="attention is all you need",
        limit=2,
    )

    assert result.source_name == "arxiv"
    assert result.papers == [fallback_paper]
    assert result.fallback_used is True
    assert len(result.failures) == 1
    assert result.failures[0].source_name == "semantic_scholar"
    assert result.failures[0].error_type == "RuntimeError"
    assert result.failures[0].message == "quota exceeded"

    assert primary_client.search_calls == [
        {
            "query": "attention is all you need",
            "limit": 2,
        }
    ]
    assert fallback_client.search_calls == [
        {
            "query": "attention is all you need",
            "limit": 2,
        }
    ]


def test_search_falls_back_when_primary_returns_empty_result() -> None:
    fallback_paper = make_paper_candidate()

    primary_client = FakePaperSearchClient(
        source_name="semantic_scholar",
        papers=[],
    )
    fallback_client = FakePaperSearchClient(
        source_name="arxiv",
        papers=[fallback_paper],
    )

    service = FallbackPaperSearchService(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )

    result = service.search(
        query="attention is all you need",
        limit=1,
    )

    assert result.source_name == "arxiv"
    assert result.papers == [fallback_paper]
    assert result.fallback_used is True
    assert len(result.failures) == 1
    assert result.failures[0].source_name == "semantic_scholar"
    assert result.failures[0].error_type == "EmptyResult"
    assert result.failures[0].message == "Primary source returned no papers."


def test_search_raises_clear_error_when_both_sources_fail() -> None:
    primary_client = FakePaperSearchClient(
        source_name="semantic_scholar",
        error=RuntimeError("quota exceeded"),
    )
    fallback_client = FakePaperSearchClient(
        source_name="arxiv",
        error=RuntimeError("network failed"),
    )

    service = FallbackPaperSearchService(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )

    with pytest.raises(
        FallbackPaperSearchError,
        match="No paper-search source returned usable results",
    ) as error_info:
        service.search(
            query="attention is all you need",
            limit=1,
        )

    error_message = str(error_info.value)

    assert "semantic_scholar: RuntimeError: quota exceeded" in error_message
    assert "arxiv: RuntimeError: network failed" in error_message


def test_search_raises_when_both_sources_return_empty_results() -> None:
    primary_client = FakePaperSearchClient(
        source_name="semantic_scholar",
        papers=[],
    )
    fallback_client = FakePaperSearchClient(
        source_name="arxiv",
        papers=[],
    )

    service = FallbackPaperSearchService(
        primary_client=primary_client,
        fallback_client=fallback_client,
    )

    with pytest.raises(
        FallbackPaperSearchError,
        match="No paper-search source returned usable results",
    ) as error_info:
        service.search(
            query="attention is all you need",
            limit=1,
        )

    error_message = str(error_info.value)

    assert "semantic_scholar: EmptyResult" in error_message
    assert "arxiv: EmptyResult" in error_message


def test_search_rejects_blank_query() -> None:
    service = FallbackPaperSearchService(
        primary_client=FakePaperSearchClient(source_name="semantic_scholar"),
        fallback_client=FakePaperSearchClient(source_name="arxiv"),
    )

    with pytest.raises(ValueError, match="Search query must not be blank"):
        service.search(
            query="   ",
            limit=1,
        )


def test_search_rejects_invalid_limit() -> None:
    service = FallbackPaperSearchService(
        primary_client=FakePaperSearchClient(source_name="semantic_scholar"),
        fallback_client=FakePaperSearchClient(source_name="arxiv"),
    )

    with pytest.raises(ValueError, match="Search limit must be at least 1"):
        service.search(
            query="attention",
            limit=0,
        )
