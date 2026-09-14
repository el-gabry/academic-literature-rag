from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from academic_literature_rag.models.research_intent import ResearchNeed


class DiscoveryQueryType(StrEnum):
    """Purpose of a query used during scientific paper discovery."""

    DIRECT = "direct"
    METHOD = "method"
    EMPIRICAL = "empirical"
    SYNTHESIS = "synthesis"
    FOUNDATIONAL = "foundational"
    BENCHMARK = "benchmark"
    LIMITATION = "limitation"
    COUNTER_EVIDENCE = "counter_evidence"
    TRANSFER = "transfer"


class DiscoveryQuery(BaseModel):
    """One purpose-specific query generated from a research intent."""

    query_type: DiscoveryQueryType
    query: str = Field(min_length=1)
    source_need: ResearchNeed | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())

        if not normalized:
            raise ValueError("Discovery query cannot be blank.")

        return normalized