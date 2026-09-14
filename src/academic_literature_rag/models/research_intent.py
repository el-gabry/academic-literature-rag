from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ResearchNeed(StrEnum):
    """Scientific information needs expressed by the researcher."""

    METHODS = "methods"
    EMPIRICAL_EVIDENCE = "empirical_evidence"
    SYNTHESIS = "synthesis"
    FOUNDATIONAL_CONTEXT = "foundational_context"
    BENCHMARKS = "benchmarks"
    LIMITATIONS = "limitations"
    COUNTER_EVIDENCE = "counter_evidence"


class ResearchIntent(BaseModel):
    """Structured representation of a researcher's scientific intent.

    This model describes what the researcher wants to investigate.
    Retrieval, source quality, corpus roles, and portfolio scoring are
    intentionally handled by later pipeline stages.
    """

    research_idea: str
    research_question: str

    target_concepts: list[str] = Field(min_length=1)
    primary_domains: list[str] = Field(min_length=1)
    secondary_domains: list[str] = Field(default_factory=list)

    research_needs: list[ResearchNeed] = Field(min_length=1)

    allow_cross_domain_transfer: bool = True

    @field_validator(
        "research_idea",
        "research_question",
    )
    @classmethod
    def validate_required_text(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate required free-text fields."""

        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Research intent text cannot be blank.")

        return normalized

    @field_validator(
        "target_concepts",
        "primary_domains",
        "secondary_domains",
    )
    @classmethod
    def normalize_text_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        """Normalize text items and remove case-insensitive duplicates."""

        normalized_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(value.split())

            if not normalized:
                continue

            normalized_key = normalized.casefold()

            if normalized_key in seen:
                continue

            seen.add(normalized_key)
            normalized_values.append(normalized)

        return normalized_values

    @field_validator("research_needs")
    @classmethod
    def deduplicate_research_needs(
        cls,
        values: list[ResearchNeed],
    ) -> list[ResearchNeed]:
        """Remove duplicate research needs while preserving order."""

        return list(dict.fromkeys(values))