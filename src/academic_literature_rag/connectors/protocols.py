from __future__ import annotations

from typing import Protocol

from academic_literature_rag.models.paper_candidate import PaperCandidate


class PaperSourceClient(Protocol):
    """Shared contract for academic-paper source connectors."""

    source_name: str
    raw_extension: str

    def fetch_raw_response(
        self,
        *,
        query: str,
        limit: int = 10,
    ) -> str:
        """Return the original source response as text."""

    def map_raw_response(
        self,
        raw_response: str,
    ) -> list[PaperCandidate]:
        """Convert a raw response into validated paper candidates."""
