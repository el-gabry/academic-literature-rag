from __future__ import annotations

import math

import pytest

from academic_literature_rag.services.retrieval_evaluation_service import (
    LabeledQuery,
    RetrievalEvaluationService,
)


class TestLabeledQuery:
    def test_create_valid(self) -> None:
        lq = LabeledQuery.create("what is attention?", ["c1", "c2"])
        assert lq.query == "what is attention?"
        assert lq.relevant_chunk_ids == frozenset({"c1", "c2"})

    def test_create_rejects_empty_query(self) -> None:
        with pytest.raises(ValueError):
            LabeledQuery.create("   ", ["c1"])

    def test_create_rejects_empty_relevant_ids(self) -> None:
        with pytest.raises(ValueError):
            LabeledQuery.create("query", [])


class TestEvaluateQuery:
    def setup_method(self) -> None:
        self.service = RetrievalEvaluationService()

    def test_perfect_retrieval_at_k(self) -> None:
        lq = LabeledQuery.create("q", ["c1", "c2"])
        result = self.service.evaluate_query(lq, ["c1", "c2"], k=2)

        assert result.precision_at_k == 1.0
        assert result.recall_at_k == 1.0
        assert result.reciprocal_rank == 1.0
        assert result.ndcg_at_k == pytest.approx(1.0)

    def test_no_relevant_results_retrieved(self) -> None:
        lq = LabeledQuery.create("q", ["c1", "c2"])
        result = self.service.evaluate_query(lq, ["c9", "c8"], k=2)

        assert result.precision_at_k == 0.0
        assert result.recall_at_k == 0.0
        assert result.reciprocal_rank == 0.0
        assert result.ndcg_at_k == 0.0

    def test_partial_hit_computes_precision_and_recall(self) -> None:
        lq = LabeledQuery.create("q", ["c1", "c2", "c3"])
        result = self.service.evaluate_query(lq, ["c1", "c9", "c3"], k=3)

        assert result.precision_at_k == pytest.approx(2 / 3)
        assert result.recall_at_k == pytest.approx(2 / 3)

    def test_reciprocal_rank_rewards_early_hit(self) -> None:
        lq = LabeledQuery.create("q", ["c1"])

        first_position = self.service.evaluate_query(lq, ["c1", "c9", "c8"], k=3)
        third_position = self.service.evaluate_query(lq, ["c9", "c8", "c1"], k=3)

        assert first_position.reciprocal_rank == 1.0
        assert third_position.reciprocal_rank == pytest.approx(1 / 3)
        assert first_position.reciprocal_rank > third_position.reciprocal_rank

    def test_ndcg_rewards_correct_ranking_order(self) -> None:
        lq = LabeledQuery.create("q", ["c1", "c2"])

        well_ranked = self.service.evaluate_query(lq, ["c1", "c2", "c9"], k=3)
        poorly_ranked = self.service.evaluate_query(lq, ["c9", "c2", "c1"], k=3)

        assert well_ranked.ndcg_at_k == pytest.approx(1.0)
        assert poorly_ranked.ndcg_at_k < well_ranked.ndcg_at_k

    def test_truncates_to_k(self) -> None:
        lq = LabeledQuery.create("q", ["c1", "c5"])
        result = self.service.evaluate_query(
            lq, ["c1", "c2", "c3", "c4", "c5"], k=2
        )

        assert result.retrieved_chunk_ids == ("c1", "c2")
        assert result.recall_at_k == pytest.approx(0.5)

    def test_rejects_non_positive_k(self) -> None:
        lq = LabeledQuery.create("q", ["c1"])
        with pytest.raises(ValueError):
            self.service.evaluate_query(lq, ["c1"], k=0)

    def test_empty_retrieved_list(self) -> None:
        lq = LabeledQuery.create("q", ["c1"])
        result = self.service.evaluate_query(lq, [], k=3)

        assert result.precision_at_k == 0.0
        assert result.recall_at_k == 0.0
        assert result.reciprocal_rank == 0.0
        assert result.ndcg_at_k == 0.0


class TestEvaluateAll:
    def setup_method(self) -> None:
        self.service = RetrievalEvaluationService()

    def test_aggregates_across_queries(self) -> None:
        queries = [
            LabeledQuery.create("q1", ["c1"]),
            LabeledQuery.create("q2", ["c2"]),
        ]
        retrieved = {
            "q1": ["c1", "c9"],
            "q2": ["c9", "c8"],
        }

        result = self.service.evaluate_all(queries, retrieved, k=2)

        assert result.query_count == 2
        assert result.mean_precision_at_k == pytest.approx((0.5 + 0.0) / 2)
        assert result.mean_recall_at_k == pytest.approx((1.0 + 0.0) / 2)
        assert len(result.per_query_results) == 2

    def test_rejects_empty_query_list(self) -> None:
        with pytest.raises(ValueError):
            self.service.evaluate_all([], {}, k=3)

    def test_raises_on_missing_retrieval_results(self) -> None:
        queries = [LabeledQuery.create("q1", ["c1"])]
        with pytest.raises(KeyError):
            self.service.evaluate_all(queries, {}, k=3)

    def test_perfect_aggregate_scores_one(self) -> None:
        queries = [
            LabeledQuery.create("q1", ["c1"]),
            LabeledQuery.create("q2", ["c2"]),
        ]
        retrieved = {
            "q1": ["c1"],
            "q2": ["c2"],
        }

        result = self.service.evaluate_all(queries, retrieved, k=1)

        assert result.mean_precision_at_k == 1.0
        assert result.mean_recall_at_k == 1.0
        assert result.mean_reciprocal_rank == 1.0
        assert result.mean_ndcg_at_k == pytest.approx(1.0)


class TestNdcgMath:
    def test_ndcg_matches_manual_calculation(self) -> None:
        service = RetrievalEvaluationService()
        lq = LabeledQuery.create("q", ["c1", "c3"])
        result = service.evaluate_query(lq, ["c9", "c1", "c3"], k=3)

        dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
        idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
        expected = dcg / idcg

        assert result.ndcg_at_k == pytest.approx(expected)
