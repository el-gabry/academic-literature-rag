from __future__ import annotations

import pytest

from academic_literature_rag.retrieval.vector_math import (
    VectorMathError,
    cosine_similarity,
)


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    similarity = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert similarity == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_orthogonal_vectors() -> None:
    similarity = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


def test_cosine_similarity_returns_negative_one_for_opposite_vectors() -> None:
    similarity = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert similarity == pytest.approx(-1.0)


def test_cosine_similarity_handles_partial_similarity() -> None:
    similarity = cosine_similarity(
        [1.0, 1.0],
        [1.0, 0.0],
    )

    assert similarity == pytest.approx(0.70710678)


def test_cosine_similarity_rejects_empty_first_vector() -> None:
    with pytest.raises(VectorMathError, match="First vector cannot be empty"):
        cosine_similarity(
            [],
            [1.0, 2.0],
        )


def test_cosine_similarity_rejects_empty_second_vector() -> None:
    with pytest.raises(VectorMathError, match="Second vector cannot be empty"):
        cosine_similarity(
            [1.0, 2.0],
            [],
        )


def test_cosine_similarity_rejects_different_dimensions() -> None:
    with pytest.raises(VectorMathError, match="same dimension"):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_cosine_similarity_rejects_zero_first_vector() -> None:
    with pytest.raises(VectorMathError, match="zero vector"):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 2.0],
        )


def test_cosine_similarity_rejects_zero_second_vector() -> None:
    with pytest.raises(VectorMathError, match="zero vector"):
        cosine_similarity(
            [1.0, 2.0],
            [0.0, 0.0],
        )
