from datetime import UTC, datetime

from academic_literature_rag.models.paper_candidate import PaperCandidate
from academic_literature_rag.models.source_assessment import (
    AssessmentConfidence,
    PublicationStatus,
    SourceAuthority,
    SourceDecision,
    TopicRelevance,
)
from academic_literature_rag.services.source_assessment_policy import (
    SourceAssessmentPolicy,
)


def build_candidate(
    *,
    source: str = "semantic_scholar",
    doi: str | None = "10.1000/example",
    arxiv_id: str | None = None,
    authors: list[str] | None = None,
    publication_year: int | None = 2026,
    venue: str | None = "Journal of Example Research",
) -> PaperCandidate:
    return PaperCandidate(
        source=source,
        source_id="paper-123",
        title="Trustworthy Change Point Detection",
        landing_url="https://example.org/paper-123",
        retrieved_at=datetime.now(UTC),
        authors=authors if authors is not None else ["A. Researcher"],
        publication_year=publication_year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
    )


def test_strong_semantic_scholar_candidate_is_accepted() -> None:
    policy = SourceAssessmentPolicy()

    candidate = build_candidate()

    assessment = policy.assess(
        candidate,
        topic_relevance=TopicRelevance.HIGH,
    )

    assert assessment.provider == "semantic_scholar"
    assert assessment.authority == SourceAuthority.AGGREGATOR
    assert assessment.publication_status == PublicationStatus.UNKNOWN
    assert assessment.metadata_confidence == AssessmentConfidence.HIGH
    assert assessment.decision == SourceDecision.ACCEPT
    assert assessment.allows_download is True
    assert assessment.requires_caution is False


def test_arxiv_candidate_is_accepted_with_caution() -> None:
    policy = SourceAssessmentPolicy()

    candidate = build_candidate(
        source="arxiv",
        doi=None,
        arxiv_id="2608.12345",
        venue=None,
    )

    assessment = policy.assess(
        candidate,
        topic_relevance=TopicRelevance.HIGH,
    )

    assert assessment.authority == SourceAuthority.PREPRINT_REPOSITORY
    assert assessment.publication_status == PublicationStatus.PREPRINT
    assert assessment.metadata_confidence == AssessmentConfidence.HIGH
    assert assessment.decision == SourceDecision.ACCEPT_WITH_CAUTION
    assert assessment.allows_download is True
    assert assessment.requires_caution is True


def test_candidate_with_weak_metadata_is_quarantined() -> None:
    policy = SourceAssessmentPolicy()

    candidate = build_candidate(
        doi=None,
        arxiv_id=None,
        authors=[],
        publication_year=None,
        venue=None,
    )

    assessment = policy.assess(
        candidate,
        topic_relevance=TopicRelevance.UNKNOWN,
    )

    assert assessment.metadata_confidence == AssessmentConfidence.LOW
    assert assessment.decision == SourceDecision.QUARANTINE
    assert assessment.allows_download is False
    assert assessment.requires_review is True


def test_irrelevant_candidate_is_rejected() -> None:
    policy = SourceAssessmentPolicy()

    candidate = build_candidate()

    assessment = policy.assess(
        candidate,
        topic_relevance=TopicRelevance.IRRELEVANT,
    )

    assert assessment.decision == SourceDecision.REJECT
    assert assessment.allows_download is False
    assert assessment.requires_review is False
    assert assessment.reasons == (
        "The paper was assessed as irrelevant to the target topic.",
    )


def test_low_relevance_semantic_scholar_candidate_is_quarantined() -> None:
    policy = SourceAssessmentPolicy()

    candidate = build_candidate()

    assessment = policy.assess(
        candidate,
        topic_relevance=TopicRelevance.LOW,
    )

    assert assessment.metadata_confidence == AssessmentConfidence.HIGH
    assert assessment.decision == SourceDecision.QUARANTINE
    assert assessment.allows_download is False
    assert assessment.requires_review is True


def test_semantic_scholar_candidate_with_medium_metadata_requires_caution() -> None:
    policy = SourceAssessmentPolicy()

    candidate = build_candidate(
        doi="10.1000/example",
        authors=[],
        publication_year=None,
        venue=None,
    )

    assessment = policy.assess(
        candidate,
        topic_relevance=TopicRelevance.MEDIUM,
    )

    assert assessment.metadata_confidence == AssessmentConfidence.MEDIUM
    assert assessment.decision == SourceDecision.ACCEPT_WITH_CAUTION
    assert assessment.allows_download is True
    assert assessment.requires_caution is True