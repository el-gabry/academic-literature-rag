from __future__ import annotations

from typing import Protocol

from academic_literature_rag.models.research_intent import ResearchIntent


class ResearchIntentParsingError(RuntimeError):
    """Raised when a research idea cannot be converted into a valid intent."""


class ResearchIntentParser(Protocol):
    """Contract for converting free-text research ideas into ResearchIntent."""

    def parse(
        self,
        research_idea: str,
    ) -> ResearchIntent:
        """Parse a free-text research idea into a structured research intent."""