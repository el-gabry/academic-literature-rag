from __future__ import annotations

from academic_literature_rag.models.discovery_query import (
    DiscoveryQuery,
    DiscoveryQueryType,
)
from academic_literature_rag.models.research_intent import (
    ResearchIntent,
    ResearchNeed,
)


class IntentQueryExpansionService:
    """Expand a structured research intent into discovery queries.

    The expansion is deterministic so that discovery behaviour is
    reproducible and suitable for evaluation against alternative query
    generation strategies.
    """

    _NEED_QUERY_MAPPING: dict[
        ResearchNeed,
        tuple[DiscoveryQueryType, tuple[str, ...]],
    ] = {
        ResearchNeed.METHODS: (
            DiscoveryQueryType.METHOD,
            (
                "methods",
                "methodology",
            ),
        ),
        ResearchNeed.EMPIRICAL_EVIDENCE: (
            DiscoveryQueryType.EMPIRICAL,
            (
                "validation",
                "empirical evaluation",
            ),
        ),
        ResearchNeed.SYNTHESIS: (
            DiscoveryQueryType.SYNTHESIS,
            (
                "review",
                "systematic review",
            ),
        ),
        ResearchNeed.FOUNDATIONAL_CONTEXT: (
            DiscoveryQueryType.FOUNDATIONAL,
            (
                "theory",
                "foundational",
            ),
        ),
        ResearchNeed.BENCHMARKS: (
            DiscoveryQueryType.BENCHMARK,
            (
                "benchmark",
                "comparative evaluation",
            ),
        ),
        ResearchNeed.LIMITATIONS: (
            DiscoveryQueryType.LIMITATION,
            (
                "limitations",
                "failure conditions",
            ),
        ),
        ResearchNeed.COUNTER_EVIDENCE: (
            DiscoveryQueryType.COUNTER_EVIDENCE,
            (
                "contradictory evidence",
                "conflicting findings",
            ),
        ),
    }

    def expand(
        self,
        intent: ResearchIntent,
    ) -> list[DiscoveryQuery]:
        """Generate purpose-specific discovery queries from an intent."""

        direct_terms = self._unique_terms(
    [
        *intent.target_concepts,
        *intent.primary_domains,
    ]
)

        anchor_terms = self._unique_terms(
            [
                intent.target_concepts[0],
                *intent.primary_domains,
            ]
        )

        queries = [
    DiscoveryQuery(
        query_type=DiscoveryQueryType.DIRECT,
        query=self._compose_query(direct_terms),
    )
        ]

        for research_need in intent.research_needs:
            query_spec = self._NEED_QUERY_MAPPING.get(
                research_need,
            )

            if query_spec is None:
                continue

            query_type, expansion_terms = query_spec

            queries.append(
                DiscoveryQuery(
                    query_type=query_type,
                    query=self._compose_query(
                        [
                            *anchor_terms,
        *expansion_terms,
                        ]
                    ),
                    source_need=research_need,
                )
            )

        if intent.allow_cross_domain_transfer:
            queries.append(
                self._build_transfer_query(intent)
            )

        return self._deduplicate_queries(queries)

    def _build_transfer_query(
        self,
        intent: ResearchIntent,
    ) -> DiscoveryQuery:
        """Create a transfer query without inventing scientific domains."""

        transfer_terms = list(intent.target_concepts)

        if intent.secondary_domains:
            transfer_terms.extend(
                intent.secondary_domains
            )
        else:
            transfer_terms.extend(
                [
                    "cross-domain",
                    "transferable methods",
                ]
            )

        return DiscoveryQuery(
            query_type=DiscoveryQueryType.TRANSFER,
            query=self._compose_query(
                self._unique_terms(transfer_terms)
            ),
        )

    @staticmethod
    def _compose_query(
        terms: list[str] | tuple[str, ...],
    ) -> str:
        """Join normalized query terms into one search string."""

        return " ".join(
            term.strip()
            for term in terms
            if term.strip()
        )

    @staticmethod
    def _unique_terms(
        terms: list[str],
    ) -> list[str]:
        """Remove duplicate terms while preserving their first occurrence."""

        unique_terms: list[str] = []
        seen: set[str] = set()

        for term in terms:
            normalized = " ".join(
                term.split()
            )

            if not normalized:
                continue

            key = normalized.casefold()

            if key in seen:
                continue

            seen.add(key)
            unique_terms.append(normalized)

        return unique_terms

    @staticmethod
    def _deduplicate_queries(
        queries: list[DiscoveryQuery],
    ) -> list[DiscoveryQuery]:
        """Remove duplicate query strings while preserving query order."""

        unique_queries: list[DiscoveryQuery] = []
        seen: set[str] = set()

        for query in queries:
            key = query.query.casefold()

            if key in seen:
                continue

            seen.add(key)
            unique_queries.append(query)

        return unique_queries