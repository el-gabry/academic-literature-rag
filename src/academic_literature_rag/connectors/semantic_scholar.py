from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx

from academic_literature_rag.models.paper_candidate import PaperCandidate


class SemanticScholarRequestError(RuntimeError):
    """Raised when Semantic Scholar cannot complete a search request."""


class SemanticScholarResponseError(RuntimeError):
    """Raised when Semantic Scholar returns an invalid response."""


class SemanticScholarClient:
    """Client responsible only for Semantic Scholar paper-search requests."""

    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    # Required by the generic persisted retrieval workflow.
    source_name = "semantic_scholar"
    raw_extension = "json"

    FIELDS = "paperId,title,url,abstract,authors,year,venue,externalIds,openAccessPdf,citationCount"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._http_client = http_client or httpx.Client(timeout=15.0)
        self._owns_http_client = http_client is None

    def close(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_http_client:
            self._http_client.close()

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperCandidate]:
        """Search Semantic Scholar and return normalized paper candidates."""

        payload = self.fetch_search_payload(
            query=query,
            limit=limit,
        )

        return self.map_search_payload(payload)

    def fetch_raw_response(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> str:
        """Fetch the original response serialized as formatted JSON text."""

        payload = self.fetch_search_payload(
            query=query,
            limit=limit,
        )

        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    def map_raw_response(
        self,
        raw_response: str,
    ) -> list[PaperCandidate]:
        """Map raw JSON text to validated paper candidates."""

        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise SemanticScholarResponseError("Semantic Scholar returned invalid JSON.") from error

        if not isinstance(payload, dict):
            raise SemanticScholarResponseError("Semantic Scholar response must be a JSON object.")

        return self.map_search_payload(payload)

    def fetch_search_payload(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Fetch the original Semantic Scholar JSON response."""

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

        if not isinstance(payload, dict):
            raise SemanticScholarResponseError("Semantic Scholar response must be a JSON object.")

        return payload

    def map_search_payload(
        self,
        payload: dict[str, Any],
    ) -> list[PaperCandidate]:
        """Map a Semantic Scholar JSON payload to paper candidates."""

        raw_papers = payload.get("data")

        if not isinstance(raw_papers, list):
            raise SemanticScholarResponseError(
                "Semantic Scholar response does not contain a valid data list."
            )

        retrieved_at = datetime.now(UTC)
        papers: list[PaperCandidate] = []

        for raw_paper in raw_papers:
            if not isinstance(raw_paper, dict):
                raise SemanticScholarResponseError(
                    "Semantic Scholar response contains an invalid paper record."
                )

            papers.append(
                self._to_paper_candidate(
                    raw_paper=raw_paper,
                    retrieved_at=retrieved_at,
                )
            )

        return papers

    @staticmethod
    def _to_paper_candidate(
        *,
        raw_paper: dict[str, Any],
        retrieved_at: datetime,
    ) -> PaperCandidate:
        """Convert one Semantic Scholar record into a PaperCandidate."""

        external_ids = raw_paper.get("externalIds") or {}
        open_access_pdf = raw_paper.get("openAccessPdf") or {}
        raw_authors = raw_paper.get("authors") or []

        if not isinstance(external_ids, dict):
            external_ids = {}

        if not isinstance(open_access_pdf, dict):
            open_access_pdf = {}

        if not isinstance(raw_authors, list):
            raw_authors = []

        authors = [
            author["name"].strip()
            for author in raw_authors
            if isinstance(author, dict)
            and isinstance(author.get("name"), str)
            and author["name"].strip()
        ]

        source_id = SemanticScholarClient._required_string(
            raw_paper,
            "paperId",
        )

        title = SemanticScholarClient._required_string(
            raw_paper,
            "title",
        )

        landing_url = SemanticScholarClient._required_string(
            raw_paper,
            "url",
        )

        return PaperCandidate(
            source="semantic_scholar",
            source_id=source_id,
            title=title,
            landing_url=landing_url,
            retrieved_at=retrieved_at,
            abstract=SemanticScholarClient._optional_string(raw_paper.get("abstract")),
            authors=authors,
            publication_year=raw_paper.get("year"),
            venue=SemanticScholarClient._optional_string(raw_paper.get("venue")),
            doi=SemanticScholarClient._optional_string(external_ids.get("DOI")),
            arxiv_id=SemanticScholarClient._optional_string(external_ids.get("ArXiv")),
            open_access_pdf_url=SemanticScholarClient._optional_string(open_access_pdf.get("url")),
            citation_count=raw_paper.get("citationCount"),
        )

    @staticmethod
    def _required_string(
        raw_paper: dict[str, Any],
        field_name: str,
    ) -> str:
        """Return a required non-empty string from an API paper record."""

        value = SemanticScholarClient._optional_string(raw_paper.get(field_name))

        if value is None:
            raise SemanticScholarResponseError(
                f"Semantic Scholar paper is missing required field: {field_name}."
            )

        return value

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        """Return a trimmed string or None."""

        if not isinstance(value, str):
            return None

        cleaned_value = value.strip()

        return cleaned_value or None
