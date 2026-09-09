from __future__ import annotations

from uuid import uuid4

import pytest

from academic_literature_rag.models.semantic_search_result import SemanticSearchResult
from academic_literature_rag.services.reranking_search_service import (
    RerankingError,
    RerankingSearchService,
)


def build_result(
    *,
    text: str,
    similarity_score: float,
) -> SemanticSearchResult:
    """Build one realistic SemanticSearchResult for test fixtures."""

    return SemanticSearchResult(
        text_chunk_id=uuid4(),
        pdf_asset_id=uuid4(),
        chunk_index=0,
        start_page_number=1,
        end_page_number=1,
        text=text,
        similarity_score=similarity_score,
        embedding_model="text-embedding-3-small",
    )


class FakeSearchService:
    """A base search service returning a fixed, ordered candidate list.

    Deliberately returns candidates in a DELIBERATELY WRONG order (worst
    match first) so tests can assert that reranking actually reorders them,
    rather than passing trivially because the input was already sorted.
    """

    def __init__(self, results: list[SemanticSearchResult]) -> None:
        self._results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SemanticSearchResult]:
        self.last_query = query
        self.last_top_k = top_k
        return self._results[:top_k]


class FakeCrossEncoder:
    """A cross-encoder stub returning caller-supplied scores in call order."""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self._scores_by_text = scores_by_text

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._scores_by_text[text] for _query, text in pairs]


def test_search_reorders_candidates_by_cross_encoder_score(monkeypatch) -> None:
    irrelevant = build_result(text="Irrelevant chunk about weather.", similarity_score=0.91)
    relevant = build_result(text="Directly answers the query.", similarity_score=0.40)

    base_service = FakeSearchService([irrelevant, relevant])
    service = RerankingSearchService(
        base_search_service=base_service,
        candidate_pool_size=20,
    )

    fake_cross_encoder = FakeCrossEncoder(
        {
            "Irrelevant chunk about weather.": 0.10,
            "Directly answers the query.": 0.95,
        }
    )
    monkeypatch.setattr(service, "_get_cross_encoder", lambda: fake_cross_encoder)

    results = service.search("What answers the query?", top_k=2)

    assert [result.text for result in results] == [
        "Directly answers the query.",
        "Irrelevant chunk about weather.",
    ]


def test_search_uses_rerank_score_as_similarity_score(monkeypatch) -> None:
    candidate = build_result(text="Some chunk.", similarity_score=0.50)

    base_service = FakeSearchService([candidate])
    service = RerankingSearchService(base_search_service=base_service)

    fake_cross_encoder = FakeCrossEncoder({"Some chunk.": 0.77})
    monkeypatch.setattr(service, "_get_cross_encoder", lambda: fake_cross_encoder)

    [result] = service.search("query", top_k=1)

    assert result.similarity_score == pytest.approx(0.77)
    assert result.embedding_model == "text-embedding-3-small+rerank"


def test_search_requests_candidate_pool_size_from_base_service(monkeypatch) -> None:
    base_service = FakeSearchService(
        [build_result(text=f"Chunk {i}.", similarity_score=0.5) for i in range(30)]
    )
    service = RerankingSearchService(
        base_search_service=base_service,
        candidate_pool_size=20,
    )

    monkeypatch.setattr(
        service,
        "_get_cross_encoder",
        lambda: FakeCrossEncoder({f"Chunk {i}.": float(i) for i in range(30)}),
    )

    service.search("query", top_k=5)

    assert base_service.last_top_k == 20


def test_search_returns_empty_list_when_base_service_has_no_candidates(monkeypatch) -> None:
    base_service = FakeSearchService([])
    service = RerankingSearchService(base_search_service=base_service)

    monkeypatch.setattr(service, "_get_cross_encoder", lambda: FakeCrossEncoder({}))

    assert service.search("query", top_k=5) == []


def test_search_rejects_empty_query() -> None:
    service = RerankingSearchService(base_search_service=FakeSearchService([]))

    with pytest.raises(RerankingError):
        service.search("   ", top_k=5)


def test_search_rejects_non_positive_top_k() -> None:
    service = RerankingSearchService(base_search_service=FakeSearchService([]))

    with pytest.raises(RerankingError):
        service.search("query", top_k=0)


def test_constructor_rejects_non_positive_candidate_pool_size() -> None:
    with pytest.raises(ValueError):
        RerankingSearchService(
            base_search_service=FakeSearchService([]),
            candidate_pool_size=0,
        )


def test_cross_encoder_is_loaded_lazily_and_cached(monkeypatch) -> None:
    load_calls: list[str] = []

    class FakeCrossEncoderClass:
        def __init__(self, model_name: str) -> None:
            load_calls.append(model_name)

        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [1.0 for _ in pairs]

    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoderClass
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    candidate = build_result(text="Chunk.", similarity_score=0.5)
    service = RerankingSearchService(base_search_service=FakeSearchService([candidate]))

    service.search("first query", top_k=1)
    service.search("second query", top_k=1)

    assert load_calls == ["cross-encoder/ms-marco-MiniLM-L-6-v2"]
