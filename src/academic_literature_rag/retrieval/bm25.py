from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenization shared by indexing and querying."""

    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Bm25Document:
    """One document registered in a BM25 index."""

    document_id: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class Bm25Match:
    """One scored BM25 result."""

    document_id: str
    score: float


class Bm25Index:
    """A small, dependency-free BM25 (Okapi) index.

    Standard BM25 formulation with k1 and b smoothing parameters. Built for
    corpora in the thousands-of-chunks range, which is the expected scale
    for a single-project academic literature RAG index. For larger corpora,
    swap this for a proper inverted-index library (e.g. rank-bm25, Tantivy).
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 < 0:
            raise ValueError("k1 must be non-negative.")

        if not (0.0 <= b <= 1.0):
            raise ValueError("b must be between 0 and 1.")

        self._k1 = k1
        self._b = b
        self._documents: dict[str, Bm25Document] = {}
        self._term_frequencies: dict[str, Counter[str]] = {}
        self._document_frequency: Counter[str] = Counter()
        self._document_lengths: dict[str, int] = {}
        self._average_document_length: float = 0.0

    @property
    def size(self) -> int:
        """Return the number of indexed documents."""

        return len(self._documents)

    def index(
        self,
        documents: list[tuple[str, str]],
    ) -> None:
        """Build the index from (document_id, text) pairs. Replaces any prior index."""

        self._documents.clear()
        self._term_frequencies.clear()
        self._document_frequency.clear()
        self._document_lengths.clear()

        for document_id, text in documents:
            tokens = tuple(tokenize(text))

            self._documents[document_id] = Bm25Document(
                document_id=document_id,
                tokens=tokens,
            )

            term_counts = Counter(tokens)
            self._term_frequencies[document_id] = term_counts
            self._document_lengths[document_id] = len(tokens)

            for term in term_counts:
                self._document_frequency[term] += 1

        if self._document_lengths:
            self._average_document_length = sum(self._document_lengths.values()) / len(
                self._document_lengths
            )
        else:
            self._average_document_length = 0.0

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[Bm25Match]:
        """Return up to top_k documents ranked by BM25 score, descending."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_terms = tokenize(query)

        if not query_terms or not self._documents:
            return []

        total_documents = len(self._documents)
        scores: dict[str, float] = {}

        for term in set(query_terms):
            document_frequency = self._document_frequency.get(term, 0)

            if document_frequency == 0:
                continue

            idf = math.log(
                1.0 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
            )

            for document_id, term_counts in self._term_frequencies.items():
                term_frequency = term_counts.get(term, 0)

                if term_frequency == 0:
                    continue

                document_length = self._document_lengths[document_id]
                length_norm = 1.0 - self._b + self._b * (
                    document_length / self._average_document_length
                    if self._average_document_length > 0
                    else 1.0
                )

                score_contribution = idf * (
                    (term_frequency * (self._k1 + 1.0))
                    / (term_frequency + self._k1 * length_norm)
                )

                scores[document_id] = scores.get(document_id, 0.0) + score_contribution

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            Bm25Match(document_id=document_id, score=score)
            for document_id, score in ranked[:top_k]
        ]