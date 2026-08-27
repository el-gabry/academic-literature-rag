from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from academic_literature_rag.models.paper_candidate import PaperCandidate


class PaperSearchClient(Protocol):
    """Protocol for paper-search clients used by fallback search."""

    source_name: str

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[PaperCandidate]:
        """Search for paper candidates."""


@dataclass(frozen=True)
class PaperSearchFailure:
    """Information about one failed paper-search source."""

    source_name: str
    error_type: str
    message: str


@dataclass(frozen=True)
class FallbackPaperSearchResult:
    """Paper-search result including source and fallback metadata."""

    source_name: str
    query: str
    papers: list[PaperCandidate]
    fallback_used: bool
    failures: tuple[PaperSearchFailure, ...]


class FallbackPaperSearchError(RuntimeError):
    """Raised when no configured paper-search source succeeds."""


class FallbackPaperSearchService:
    """Search papers with a primary source and fallback source."""

    def __init__(
        self,
        *,
        primary_client: PaperSearchClient,
        fallback_client: PaperSearchClient,
    ) -> None:
        self._primary_client = primary_client
        self._fallback_client = fallback_client

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> FallbackPaperSearchResult:
        """Search primary source first, then fallback source if needed."""

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("Search query must not be blank.")

        if limit < 1:
            raise ValueError("Search limit must be at least 1.")

        failures: list[PaperSearchFailure] = []

        primary_papers = self._search_primary_source(
            query=cleaned_query,
            limit=limit,
            failures=failures,
        )

        if primary_papers:
            return FallbackPaperSearchResult(
                source_name=self._primary_client.source_name,
                query=cleaned_query,
                papers=primary_papers,
                fallback_used=False,
                failures=tuple(failures),
            )

        fallback_papers = self._search_fallback_source(
            query=cleaned_query,
            limit=limit,
            failures=failures,
        )

        if fallback_papers:
            return FallbackPaperSearchResult(
                source_name=self._fallback_client.source_name,
                query=cleaned_query,
                papers=fallback_papers,
                fallback_used=True,
                failures=tuple(failures),
            )

        raise FallbackPaperSearchError(
            self._format_failures(failures)
        )

    def _search_primary_source(
        self,
        *,
        query: str,
        limit: int,
        failures: list[PaperSearchFailure],
    ) -> list[PaperCandidate]:
        failure_count_before_search = len(failures)

        papers = self._search_source(
            client=self._primary_client,
            query=query,
            limit=limit,
            failures=failures,
        )

        if papers:
            return papers

        if len(failures) == failure_count_before_search:
            failures.append(
                PaperSearchFailure(
                    source_name=self._primary_client.source_name,
                    error_type="EmptyResult",
                    message="Primary source returned no papers.",
                )
            )

        return []

    def _search_fallback_source(
        self,
        *,
        query: str,
        limit: int,
        failures: list[PaperSearchFailure],
    ) -> list[PaperCandidate]:
        failure_count_before_search = len(failures)

        papers = self._search_source(
            client=self._fallback_client,
            query=query,
            limit=limit,
            failures=failures,
        )

        if papers:
            return papers

        if len(failures) == failure_count_before_search:
            failures.append(
                PaperSearchFailure(
                    source_name=self._fallback_client.source_name,
                    error_type="EmptyResult",
                    message="Fallback source returned no papers.",
                )
            )

        return []

    @staticmethod
    def _search_source(
        *,
        client: PaperSearchClient,
        query: str,
        limit: int,
        failures: list[PaperSearchFailure],
    ) -> list[PaperCandidate]:
        try:
            return list(
                client.search(
                    query=query,
                    limit=limit,
                )
            )
        except Exception as error:
            failures.append(
                PaperSearchFailure(
                    source_name=client.source_name,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )

            return []

    @staticmethod
    def _format_failures(
        failures: Sequence[PaperSearchFailure],
    ) -> str:
        failure_lines = [
            (
                f"- {failure.source_name}: "
                f"{failure.error_type}: {failure.message}"
            )
            for failure in failures
        ]

        return (
            "No paper-search source returned usable results.\n"
            + "\n".join(failure_lines)
        )