from __future__ import annotations

from datetime import UTC, datetime

from academic_literature_rag.identity.matching import (
    PaperMatchKind,
    compare_paper_candidates,
)
from academic_literature_rag.models.paper_candidate import PaperCandidate


def make_candidate(
    *,
    source: str,
    source_id: str,
    title: str = "Early Warning Signals in Clinical Time Series",
    doi: str | None = None,
    arxiv_id: str | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        source=source,
        source_id=source_id,
        title=title,
        landing_url=f"https://example.org/papers/{source_id}",
        retrieved_at=datetime.now(UTC),
        doi=doi,
        arxiv_id=arxiv_id,
    )


def test_matching_doi_is_safe_for_automatic_merge() -> None:
    left = make_candidate(
        source="semantic_scholar",
        source_id="semantic-1",
        doi="https://doi.org/10.1000/ABC.123",
    )

    right = make_candidate(
        source="arxiv",
        source_id="arxiv-1",
        doi="doi:10.1000/abc.123",
    )

    decision = compare_paper_candidates(left, right)

    assert decision.kind == PaperMatchKind.EXACT_DOI
    assert decision.can_auto_merge is True


def test_matching_arxiv_id_is_safe_for_automatic_merge() -> None:
    left = make_candidate(
        source="semantic_scholar",
        source_id="semantic-1",
        arxiv_id="arXiv:2302.01204v2",
    )

    right = make_candidate(
        source="arxiv",
        source_id="2302.01204v1",
        arxiv_id="2302.01204v1",
    )

    decision = compare_paper_candidates(left, right)

    assert decision.kind == PaperMatchKind.EXACT_ARXIV_ID
    assert decision.can_auto_merge is True


def test_same_title_without_strong_identifier_requires_review() -> None:
    left = make_candidate(
        source="semantic_scholar",
        source_id="semantic-1",
    )

    right = make_candidate(
        source="arxiv",
        source_id="arxiv-1",
    )

    decision = compare_paper_candidates(left, right)

    assert decision.kind == PaperMatchKind.TITLE_CANDIDATE
    assert decision.can_auto_merge is False


def test_conflicting_strong_identifiers_are_not_merged() -> None:
    left = make_candidate(
        source="semantic_scholar",
        source_id="semantic-1",
        doi="10.1000/shared-doi",
        arxiv_id="2302.01204v1",
    )

    right = make_candidate(
        source="arxiv",
        source_id="arxiv-1",
        doi="10.1000/shared-doi",
        arxiv_id="2401.99999v2",
    )

    decision = compare_paper_candidates(left, right)

    assert decision.kind == PaperMatchKind.CONFLICT
    assert decision.can_auto_merge is False


def test_unrelated_papers_do_not_match() -> None:
    left = make_candidate(
        source="semantic_scholar",
        source_id="semantic-1",
        title="Bayesian Online Change Point Detection",
        doi="10.1000/first",
    )

    right = make_candidate(
        source="arxiv",
        source_id="arxiv-1",
        title="Deep Learning for Medical Images",
        doi="10.1000/second",
    )

    decision = compare_paper_candidates(left, right)

    assert decision.kind == PaperMatchKind.NO_MATCH
    assert decision.can_auto_merge is False
