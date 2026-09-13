from academic_literature_rag.models.source_assessment import (
    AssessmentConfidence,
    EvidenceType,
    PeerReviewStatus,
    PublicationStatus,
    RetractionStatus,
    SourceAssessment,
    SourceAuthority,
    SourceDecision,
    TopicRelevance,
)


def build_assessment(
    decision: SourceDecision,
) -> SourceAssessment:
    return SourceAssessment(
        provider="semantic_scholar",
        authority=SourceAuthority.CURATED_INDEX,
        publication_status=PublicationStatus.PUBLISHED,
        peer_review_status=PeerReviewStatus.VERIFIED,
        evidence_type=EvidenceType.METHOD_PAPER,
        retraction_status=RetractionStatus.NOT_IDENTIFIED,
        metadata_confidence=AssessmentConfidence.HIGH,
        topic_relevance=TopicRelevance.HIGH,
        decision=decision,
        reasons=("Test assessment.",),
    )


def test_accept_allows_download_without_caution() -> None:
    assessment = build_assessment(SourceDecision.ACCEPT)

    assert assessment.allows_download is True
    assert assessment.requires_caution is False
    assert assessment.requires_review is False


def test_accept_with_caution_allows_download_and_requires_caution() -> None:
    assessment = build_assessment(SourceDecision.ACCEPT_WITH_CAUTION)

    assert assessment.allows_download is True
    assert assessment.requires_caution is True
    assert assessment.requires_review is False


def test_quarantine_blocks_download_and_requires_review() -> None:
    assessment = build_assessment(SourceDecision.QUARANTINE)

    assert assessment.allows_download is False
    assert assessment.requires_caution is False
    assert assessment.requires_review is True


def test_reject_blocks_download_without_review_queue() -> None:
    assessment = build_assessment(SourceDecision.REJECT)

    assert assessment.allows_download is False
    assert assessment.requires_caution is False
    assert assessment.requires_review is False