from __future__ import annotations

import math


class VectorMathError(ValueError):
    """Raised when vector similarity cannot be computed."""


def cosine_similarity(
    first_vector: list[float],
    second_vector: list[float],
) -> float:
    """Compute cosine similarity between two vectors."""

    _validate_vectors(
        first_vector=first_vector,
        second_vector=second_vector,
    )

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(
            first_vector,
            second_vector,
            strict=True,
        )
    )

    first_magnitude = _vector_magnitude(first_vector)
    second_magnitude = _vector_magnitude(second_vector)

    if first_magnitude == 0 or second_magnitude == 0:
        raise VectorMathError("Cosine similarity cannot be computed for a zero vector.")

    return dot_product / (first_magnitude * second_magnitude)


def _validate_vectors(
    *,
    first_vector: list[float],
    second_vector: list[float],
) -> None:
    """Validate vectors before similarity calculation."""

    if not first_vector:
        raise VectorMathError("First vector cannot be empty.")

    if not second_vector:
        raise VectorMathError("Second vector cannot be empty.")

    if len(first_vector) != len(second_vector):
        raise VectorMathError("Vectors must have the same dimension.")


def _vector_magnitude(
    vector: list[float],
) -> float:
    """Return the Euclidean magnitude of a vector."""

    return math.sqrt(sum(value * value for value in vector))
