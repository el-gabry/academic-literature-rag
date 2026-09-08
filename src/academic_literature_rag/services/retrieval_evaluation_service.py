from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabeledQuery:
    """A single evaluation query with its ground-truth relevant chunk ids.

    relevant_chunk_ids should contain every chunk id considered relevant
    for this query, regardless of rank. Order does not matter here.
    """

    query: str
    relevant_chunk_ids: frozenset[str]

    @staticmethod
    def create(query: str, relevant_chunk_ids: list[str]) -> "LabeledQuery":
        if not query.strip():
            raise ValueError("query must not be empty")
        if not relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty")
        return LabeledQuery(query=query, relevant_chunk_ids=frozenset(relevant_chunk_ids))


@dataclass(frozen=True)
class QueryEvaluationResult:
    """Per-query retrieval metrics at a fixed cutoff k."""

    query: str
    k: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved_chunk_ids: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AggregateEvaluationResult:
    """Aggregate retrieval metrics across all evaluated queries."""

    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    query_count: int
    per_query_results: tuple[QueryEvaluationResult, ...]


class RetrievalEvaluationService:
    """Computes standard IR metrics for a retrieval method against labeled queries.

    This service is retrieval-method agnostic: it accepts already-ranked
    retrieved chunk ids for a query and compares them against ground-truth
    relevant chunk ids. Callers run dense-only and hybrid retrieval
    separately, then pass each result set here to compare methods on
    identical queries and cutoff k.
    """

    def evaluate_query(
        self,
        labeled_query: LabeledQuery,
        retrieved_chunk_ids: list[str],
        *,
        k: int,
    ) -> QueryEvaluationResult:
        """Score a single query's retrieved results against ground truth."""
        if k <= 0:
            raise ValueError("k must be a positive integer")

        top_k = tuple(retrieved_chunk_ids[:k])
        relevant = labeled_query.relevant_chunk_ids

        precision = self._precision_at_k(top_k, relevant)
        recall = self._recall_at_k(top_k, relevant)
        rr = self._reciprocal_rank(top_k, relevant)
        ndcg = self._ndcg_at_k(top_k, relevant)

        return QueryEvaluationResult(
            query=labeled_query.query,
            k=k,
            precision_at_k=precision,
            recall_at_k=recall,
            reciprocal_rank=rr,
            ndcg_at_k=ndcg,
            retrieved_chunk_ids=top_k,
        )

    def evaluate_all(
        self,
        labeled_queries: list[LabeledQuery],
        retrieved_chunk_ids_by_query: dict[str, list[str]],
        *,
        k: int,
    ) -> AggregateEvaluationResult:
        """Score many queries and aggregate metrics with simple averaging.

        retrieved_chunk_ids_by_query must contain one entry per labeled
        query's `query` text, mapping to that query's ranked retrieval
        results.
        """
        if not labeled_queries:
            raise ValueError("labeled_queries must not be empty")

        per_query_results: list[QueryEvaluationResult] = []
        for labeled_query in labeled_queries:
            if labeled_query.query not in retrieved_chunk_ids_by_query:
                raise KeyError(
                    f"Missing retrieval results for query: {labeled_query.query!r}"
                )
            retrieved = retrieved_chunk_ids_by_query[labeled_query.query]
            per_query_results.append(self.evaluate_query(labeled_query, retrieved, k=k))

        count = len(per_query_results)
        mean_precision = sum(r.precision_at_k for r in per_query_results) / count
        mean_recall = sum(r.recall_at_k for r in per_query_results) / count
        mean_rr = sum(r.reciprocal_rank for r in per_query_results) / count
        mean_ndcg = sum(r.ndcg_at_k for r in per_query_results) / count

        return AggregateEvaluationResult(
            k=k,
            mean_precision_at_k=mean_precision,
            mean_recall_at_k=mean_recall,
            mean_reciprocal_rank=mean_rr,
            mean_ndcg_at_k=mean_ndcg,
            query_count=count,
            per_query_results=tuple(per_query_results),
        )

    @staticmethod
    def _precision_at_k(
        top_k: tuple[str, ...], relevant: frozenset[str]
    ) -> float:
        if not top_k:
            return 0.0
        hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
        return hits / len(top_k)

    @staticmethod
    def _recall_at_k(
        top_k: tuple[str, ...], relevant: frozenset[str]
    ) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for chunk_id in top_k if chunk_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def _reciprocal_rank(
        top_k: tuple[str, ...], relevant: frozenset[str]
    ) -> float:
        for rank, chunk_id in enumerate(top_k, start=1):
            if chunk_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _ndcg_at_k(
        top_k: tuple[str, ...], relevant: frozenset[str]
    ) -> float:
        if not relevant or not top_k:
            return 0.0

        dcg = 0.0
        for rank, chunk_id in enumerate(top_k, start=1):
            gain = 1.0 if chunk_id in relevant else 0.0
            if gain:
                dcg += gain / math.log2(rank + 1)

        ideal_hits = min(len(relevant), len(top_k))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg
