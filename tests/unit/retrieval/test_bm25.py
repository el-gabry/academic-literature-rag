from __future__ import annotations

import pytest

from academic_literature_rag.retrieval.bm25 import Bm25Index


def build_index() -> Bm25Index:
    index = Bm25Index()

    index.index(
        [
            (
                "chunk_1",
                "The Transformer uses self-attention mechanisms for sequence modeling.",
            ),
            (
                "chunk_2",
                "Recurrent neural networks process sequences step by step.",
            ),
            (
                "chunk_3",
                "Attention and recurrent models are both used in deep learning for NLP.",
            ),
            (
                "chunk_4",
                "Bayesian online changepoint detection identifies abrupt shifts in time series.",
            ),
        ]
    )

    return index


def test_search_ranks_exact_term_matches_above_partial_matches() -> None:
    index = build_index()

    results = index.search(
        "attention mechanism",
        top_k=4,
    )

    assert [result.document_id for result in results] == ["chunk_1", "chunk_3"]
    assert results[0].score > results[1].score


def test_search_finds_exact_terminology_dense_search_would_miss() -> None:
    index = build_index()

    results = index.search(
        "Bayesian changepoint",
        top_k=4,
    )

    assert len(results) == 1
    assert results[0].document_id == "chunk_4"


def test_search_respects_top_k() -> None:
    index = build_index()

    results = index.search(
        "recurrent",
        top_k=1,
    )

    assert len(results) == 1


def test_search_returns_empty_list_for_no_matching_terms() -> None:
    index = build_index()

    assert index.search("nonexistent term zzz", top_k=3) == []


def test_search_returns_empty_list_for_empty_query() -> None:
    index = build_index()

    assert index.search("   ", top_k=3) == []


def test_search_returns_empty_list_when_index_is_empty() -> None:
    index = Bm25Index()
    index.index([])

    assert index.search("attention", top_k=3) == []


def test_search_rejects_invalid_top_k() -> None:
    index = build_index()

    with pytest.raises(ValueError, match="top_k"):
        index.search("attention", top_k=0)


def test_index_rejects_invalid_k1() -> None:
    with pytest.raises(ValueError, match="k1"):
        Bm25Index(k1=-1.0)


def test_index_rejects_invalid_b() -> None:
    with pytest.raises(ValueError, match="b must be"):
        Bm25Index(b=1.5)


def test_index_reports_size() -> None:
    index = build_index()

    assert index.size == 4


def test_reindexing_replaces_previous_index() -> None:
    index = build_index()
    index.index([("chunk_5", "Only one document now.")])

    assert index.size == 1
    assert index.search("attention", top_k=3) == []