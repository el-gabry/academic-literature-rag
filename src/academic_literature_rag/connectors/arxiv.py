from __future__ import annotations

import re
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

import httpx

from academic_literature_rag.models.paper_candidate import PaperCandidate


class ArxivRequestError(RuntimeError):
    """Raised when arXiv cannot complete a search request."""


class ArxivResponseError(RuntimeError):
    """Raised when arXiv returns an invalid Atom XML response."""


class ArxivClient:
    """Client responsible only for arXiv paper-search requests."""

    SEARCH_URL = "http://export.arxiv.org/api/query"

    NAMESPACES = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._http_client = http_client or httpx.Client(timeout=15.0)
        self._owns_http_client = http_client is None

    def close(self) -> None:
        """Close the internally created HTTP client."""

        if self._owns_http_client:
            self._http_client.close()

    def search(
        self,
        *,
        query: str,
        limit: int = 10,
        start: int = 0,
    ) -> list[PaperCandidate]:
        """Search arXiv and return validated paper candidates."""

        feed_xml = self.fetch_search_feed(
            query=query,
            limit=limit,
            start=start,
        )

        return self.map_search_feed(feed_xml)

    def fetch_search_feed(
        self,
        *,
        query: str,
        limit: int = 10,
        start: int = 0,
    ) -> str:
        """Fetch the original Atom XML response from arXiv."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Search query must not be blank.")

        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        if start < 0:
            raise ValueError("Search start must not be negative.")

        params = {
            "search_query": f'all:"{cleaned_query}"',
            "start": start,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            response = self._http_client.get(
                self.SEARCH_URL,
                params=params,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ArxivRequestError("arXiv search request failed.") from error

        return response.text

    def map_search_feed(self, feed_xml: str) -> list[PaperCandidate]:
        """Map an Atom XML response to validated paper candidates."""

        try:
            root = ET.fromstring(feed_xml)
        except ET.ParseError as error:
            raise ArxivResponseError("arXiv returned invalid Atom XML.") from error

        retrieved_at = datetime.now(UTC)

        return [
            self._to_paper_candidate(
                entry=entry,
                retrieved_at=retrieved_at,
            )
            for entry in root.findall("atom:entry", self.NAMESPACES)
        ]

    def _to_paper_candidate(
        self,
        *,
        entry: ET.Element,
        retrieved_at: datetime,
    ) -> PaperCandidate:
        entry_url = self._required_text(
            entry,
            "atom:id",
            "entry ID",
        )

        versioned_arxiv_id = entry_url.rstrip("/").rsplit("/", maxsplit=1)[-1]

        arxiv_id = re.sub(
            r"v\d+$",
            "",
            versioned_arxiv_id,
        )

        title = self._required_text(
            entry,
            "atom:title",
            "title",
        )

        authors = [
            self._normalize_text(author_name.text)
            for author_name in entry.findall(
                "atom:author/atom:name",
                self.NAMESPACES,
            )
            if author_name.text
        ]

        pdf_url = self._find_pdf_url(entry)

        return PaperCandidate(
            source="arxiv",
            source_id=versioned_arxiv_id,
            title=self._normalize_text(title),
            landing_url=entry_url,
            retrieved_at=retrieved_at,
            abstract=self._optional_text(entry, "atom:summary"),
            authors=authors,
            publication_year=self._published_year(entry),
            venue=self._optional_text(entry, "arxiv:journal_ref"),
            doi=self._optional_text(entry, "arxiv:doi"),
            arxiv_id=arxiv_id,
            open_access_pdf_url=pdf_url,
            citation_count=None,
        )

    def _published_year(self, entry: ET.Element) -> int | None:
        published_text = self._optional_text(entry, "atom:published")

        if published_text is None:
            return None

        try:
            return datetime.fromisoformat(published_text.replace("Z", "+00:00")).year
        except ValueError as error:
            raise ArxivResponseError("arXiv returned an invalid published date.") from error

    def _find_pdf_url(self, entry: ET.Element) -> str | None:
        for link in entry.findall("atom:link", self.NAMESPACES):
            if link.get("type") == "application/pdf":
                return link.get("href")

        return None

    def _required_text(
        self,
        element: ET.Element,
        path: str,
        field_name: str,
    ) -> str:
        value = self._optional_text(element, path)

        if value is None:
            raise ArxivResponseError(f"arXiv entry is missing required field: {field_name}.")

        return value

    def _optional_text(
        self,
        element: ET.Element,
        path: str,
    ) -> str | None:
        child = element.find(path, self.NAMESPACES)

        if child is None or child.text is None:
            return None

        normalized_value = self._normalize_text(child.text)

        return normalized_value or None

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.split())
