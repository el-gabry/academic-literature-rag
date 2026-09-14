from __future__ import annotations

import pytest
from pydantic import ValidationError

from academic_literature_rag.models.discovery_query import (
    DiscoveryQuery,
    DiscoveryQueryType,
)
from academic_literature_rag.models.research_intent import ResearchNeed


def test_normalizes_query_text() -> None:
    query = DiscoveryQuery(
        query_type=DiscoveryQueryType.DIRECT,
        query="  change-point   detection   physiological time series  ",
    )

    assert query.query == (
        "change-point detection physiological time series"
    )


def test_allows_direct_query_without_source_need() -> None:
    query = DiscoveryQuery(
        query_type=DiscoveryQueryType.DIRECT,
        query="change-point detection physiological time series",
    )

    assert query.query_type is DiscoveryQueryType.DIRECT
    assert query.source_need is None


def test_preserves_source_need_provenance() -> None:
    query = DiscoveryQuery(
        query_type=DiscoveryQueryType.COUNTER_EVIDENCE,
        query=(
            "change-point detection physiological time series "
            "contradictory evidence"
        ),
        source_need=ResearchNeed.COUNTER_EVIDENCE,
    )

    assert query.source_need is ResearchNeed.COUNTER_EVIDENCE


@pytest.mark.parametrize(
    "query_type",
    list(DiscoveryQueryType),
)
def test_supports_all_discovery_query_types(
    query_type: DiscoveryQueryType,
) -> None:
    query = DiscoveryQuery(
        query_type=query_type,
        query="change-point detection",
    )

    assert query.query_type is query_type


def test_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        DiscoveryQuery(
            query_type=DiscoveryQueryType.DIRECT,
            query="   ",
        )


def test_rejects_invalid_query_type() -> None:
    with pytest.raises(ValidationError):
        DiscoveryQuery(
            query_type="unknown",
            query="change-point detection",
        )


def test_rejects_invalid_source_need() -> None:
    with pytest.raises(ValidationError):
        DiscoveryQuery(
            query_type=DiscoveryQueryType.METHOD,
            query="change-point detection methods",
            source_need="unknown",
        )