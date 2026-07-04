from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from academic_literature_rag.models.paper_candidate import PaperCandidate


class SemanticScholarRequestError(RuntimeError):
    """Raised when Semantic Scholar cannot complete a search request."""


class SemanticScholarResponseError(RuntimeError):
    """Raised when Semantic Scholar returns an unexpected response format."""


class SemanticScholarClient:
    """Client responsible only for Semantic Scholar paper-search requests."""

    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    FIELDS = "title,url,abstract,authors,year,venue,externalIds,openAccessPdf,citationCount"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client or httpx.Client(timeout=10.0)
        self._owns_http_client = http_client is None

    def close(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_http_client:
            self._http_client.close()

    def search(self, query: str, limit: int = 10) -> list[PaperCandidate]:
        """Search Semantic Scholar and map results to PaperCandidate objects."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Search query must not be blank.")

        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        headers: dict[str, str] = {}

        if self._api_key:
            headers["x-api-key"] = self._api_key

        params = {
            "query": cleaned_query,
            "limit": limit,
            "fields": self.FIELDS,
        }

        try:
            response = self._http_client.get(
                self.SEARCH_URL,
                params=params,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise SemanticScholarRequestError("Semantic Scholar search request failed.") from error

        try:
            payload = response.json()
        except ValueError as error:
            raise SemanticScholarResponseError("Semantic Scholar returned invalid JSON.") from error

        raw_papers = payload.get("data")

        if not isinstance(raw_papers, list):
            raise SemanticScholarResponseError(
                "Semantic Scholar response does not contain a valid data list."
            )

        retrieved_at = datetime.now(UTC)

        return [self._to_paper_candidate(raw_paper, retrieved_at) for raw_paper in raw_papers]

    @staticmethod
    def _to_paper_candidate(
        raw_paper: dict[str, Any],
        retrieved_at: datetime,
    ) -> PaperCandidate:
        external_ids = raw_paper.get("externalIds") or {}
        open_access_pdf = raw_paper.get("openAccessPdf") or {}
        raw_authors = raw_paper.get("authors") or []

        authors = [
            author["name"]
            for author in raw_authors
            if isinstance(author, dict) and author.get("name")
        ]

        return PaperCandidate(
            source="semantic_scholar",
            source_id=raw_paper["paperId"],
            title=raw_paper["title"],
            landing_url=raw_paper["url"],
            retrieved_at=retrieved_at,
            abstract=raw_paper.get("abstract"),
            authors=authors,
            publication_year=raw_paper.get("year"),
            venue=raw_paper.get("venue"),
            doi=external_ids.get("DOI"),
            open_access_pdf_url=open_access_pdf.get("url"),
            citation_count=raw_paper.get("citationCount"),
        )
