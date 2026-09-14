from __future__ import annotations

from academic_literature_rag.models.discovery_query import (
    DiscoveryQueryType,
)
from academic_literature_rag.models.research_intent import (
    ResearchIntent,
    ResearchNeed,
)
from academic_literature_rag.services.intent_query_expansion_service import (
    IntentQueryExpansionService,
)


def build_intent(
    *,
    research_needs: list[ResearchNeed],
    allow_cross_domain_transfer: bool = False,
    secondary_domains: list[str] | None = None,
) -> ResearchIntent:
    return ResearchIntent(
        research_idea=(
            "I want to investigate reliable change-point detection "
            "in physiological time series."
        ),
        research_question=(
            "How can change-point detection be validated reliably?"
        ),
        target_concepts=[
            "change-point detection",
            "reliability",
            "uncertainty",
            "robustness",
        ],
        primary_domains=[
            "physiological time series",
        ],
        secondary_domains=secondary_domains or [],
        research_needs=research_needs,
        allow_cross_domain_transfer=allow_cross_domain_transfer,
    )


def test_always_generates_direct_query() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    assert queries[0].query_type is DiscoveryQueryType.DIRECT
    assert queries[0].source_need is None
    assert queries[0].query == (
        "change-point detection reliability uncertainty robustness "
        "physiological time series"
    )

def test_specialized_queries_use_anchor_concept_not_all_concepts() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.BENCHMARKS,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    benchmark_query = queries[1]

    assert "change-point detection" in benchmark_query.query
    assert "physiological time series" in benchmark_query.query
    assert "benchmark" in benchmark_query.query

    assert "reliability" not in benchmark_query.query
    assert "uncertainty" not in benchmark_query.query
    assert "robustness" not in benchmark_query.query
def test_generates_queries_only_for_requested_needs() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
            ResearchNeed.BENCHMARKS,
            ResearchNeed.COUNTER_EVIDENCE,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    assert [query.query_type for query in queries] == [
        DiscoveryQueryType.DIRECT,
        DiscoveryQueryType.METHOD,
        DiscoveryQueryType.BENCHMARK,
        DiscoveryQueryType.COUNTER_EVIDENCE,
    ]


def test_preserves_research_need_provenance() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
            ResearchNeed.EMPIRICAL_EVIDENCE,
            ResearchNeed.SYNTHESIS,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    specialized_queries = queries[1:]

    assert [
        query.source_need
        for query in specialized_queries
    ] == [
        ResearchNeed.METHODS,
        ResearchNeed.EMPIRICAL_EVIDENCE,
        ResearchNeed.SYNTHESIS,
    ]


def test_generates_empirical_query_terms() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.EMPIRICAL_EVIDENCE,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    empirical_query = queries[1]

    assert empirical_query.query_type is DiscoveryQueryType.EMPIRICAL
    assert "validation" in empirical_query.query
    assert "empirical evaluation" in empirical_query.query


def test_generates_counter_evidence_query_terms() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.COUNTER_EVIDENCE,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    counter_query = queries[1]

    assert (
        counter_query.query_type
        is DiscoveryQueryType.COUNTER_EVIDENCE
    )
    assert "contradictory evidence" in counter_query.query
    assert "conflicting findings" in counter_query.query


def test_does_not_generate_transfer_query_when_disabled() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
        ],
        allow_cross_domain_transfer=False,
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    assert all(
        query.query_type is not DiscoveryQueryType.TRANSFER
        for query in queries
    )


def test_transfer_query_relaxes_primary_domain_constraint() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
        ],
        allow_cross_domain_transfer=True,
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    transfer_query = next(
        query
        for query in queries
        if query.query_type is DiscoveryQueryType.TRANSFER
    )

    assert "physiological time series" not in transfer_query.query
    assert "change-point detection" in transfer_query.query
    assert "cross-domain" in transfer_query.query
    assert "transferable methods" in transfer_query.query


def test_transfer_query_uses_explicit_secondary_domains() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
        ],
        allow_cross_domain_transfer=True,
        secondary_domains=[
            "financial time series",
            "industrial sensor data",
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    transfer_query = next(
        query
        for query in queries
        if query.query_type is DiscoveryQueryType.TRANSFER
    )

    assert "financial time series" in transfer_query.query
    assert "industrial sensor data" in transfer_query.query
    assert "physiological time series" not in transfer_query.query
    assert "cross-domain" not in transfer_query.query


def test_preserves_research_need_order() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.SYNTHESIS,
            ResearchNeed.LIMITATIONS,
            ResearchNeed.BENCHMARKS,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    assert [query.query_type for query in queries] == [
        DiscoveryQueryType.DIRECT,
        DiscoveryQueryType.SYNTHESIS,
        DiscoveryQueryType.LIMITATION,
        DiscoveryQueryType.BENCHMARK,
    ]


def test_removes_duplicate_base_terms() -> None:
    intent = ResearchIntent(
        research_idea="Study reliable change-point detection.",
        research_question="How reliable is change-point detection?",
        target_concepts=[
            "change-point detection",
            "reliability",
            "Reliability",
        ],
        primary_domains=[
            "physiological time series",
            "Physiological Time Series",
        ],
        secondary_domains=[],
        research_needs=[
            ResearchNeed.METHODS,
        ],
        allow_cross_domain_transfer=False,
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    direct_query = queries[0].query.casefold()

    assert direct_query.count("reliability") == 1
    assert direct_query.count("physiological time series") == 1


def test_supports_all_research_need_query_mappings() -> None:
    intent = build_intent(
        research_needs=[
            ResearchNeed.METHODS,
            ResearchNeed.EMPIRICAL_EVIDENCE,
            ResearchNeed.SYNTHESIS,
            ResearchNeed.FOUNDATIONAL_CONTEXT,
            ResearchNeed.BENCHMARKS,
            ResearchNeed.LIMITATIONS,
            ResearchNeed.COUNTER_EVIDENCE,
        ],
    )

    service = IntentQueryExpansionService()

    queries = service.expand(intent)

    assert [query.query_type for query in queries] == [
        DiscoveryQueryType.DIRECT,
        DiscoveryQueryType.METHOD,
        DiscoveryQueryType.EMPIRICAL,
        DiscoveryQueryType.SYNTHESIS,
        DiscoveryQueryType.FOUNDATIONAL,
        DiscoveryQueryType.BENCHMARK,
        DiscoveryQueryType.LIMITATION,
        DiscoveryQueryType.COUNTER_EVIDENCE,
    ]