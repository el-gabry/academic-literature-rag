from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceAuthority(StrEnum):
    """Describes the authority of the source that exposed the document."""

    OFFICIAL_REPOSITORY = "official_repository"
    CURATED_INDEX = "curated_index"
    PUBLISHER = "publisher"
    PREPRINT_REPOSITORY = "preprint_repository"
    AGGREGATOR = "aggregator"
    UNKNOWN = "unknown"


class PublicationStatus(StrEnum):
    """Describes the known publication state of a scientific document."""

    PUBLISHED = "published"
    ACCEPTED = "accepted"
    PREPRINT = "preprint"
    WITHDRAWN = "withdrawn"
    RETRACTED = "retracted"
    UNKNOWN = "unknown"


class PeerReviewStatus(StrEnum):
    """Describes whether peer-review status can be established."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class EvidenceType(StrEnum):
    """High-level scientific evidence or document type."""

    GUIDELINE = "guideline"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    RANDOMIZED_TRIAL = "randomized_trial"
    OBSERVATIONAL_STUDY = "observational_study"
    METHOD_PAPER = "method_paper"
    BENCHMARK_PAPER = "benchmark_paper"
    REVIEW = "review"
    POSITION_STATEMENT = "position_statement"
    OTHER = "other"
    UNKNOWN = "unknown"


class RetractionStatus(StrEnum):
    """Known retraction or publication-integrity status."""

    NOT_IDENTIFIED = "not_identified"
    RETRACTED = "retracted"
    EXPRESSION_OF_CONCERN = "expression_of_concern"
    UNKNOWN = "unknown"


class AssessmentConfidence(StrEnum):
    """Confidence in a derived assessment dimension."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class TopicRelevance(StrEnum):
    """Estimated relationship between a document and the target topic."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class SourceDecision(StrEnum):
    """Decision made before a document enters the active PDF corpus."""

    ACCEPT = "accept"
    ACCEPT_WITH_CAUTION = "accept_with_caution"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass(frozen=True)
class SourceAssessment:
    """Explainable assessment of a candidate scientific source.

    This object deliberately avoids collapsing source quality into one
    numerical trust score. Each dimension remains visible so later
    retrieval, reporting, and validation logic can explain why a source
    was accepted or rejected.
    """

    provider: str
    authority: SourceAuthority
    publication_status: PublicationStatus
    peer_review_status: PeerReviewStatus
    evidence_type: EvidenceType
    retraction_status: RetractionStatus
    metadata_confidence: AssessmentConfidence
    topic_relevance: TopicRelevance
    decision: SourceDecision
    reasons: tuple[str, ...]

    @property
    def allows_download(self) -> bool:
        """Return whether the source may proceed to PDF download."""

        return self.decision in {
            SourceDecision.ACCEPT,
            SourceDecision.ACCEPT_WITH_CAUTION,
        }

    @property
    def requires_caution(self) -> bool:
        """Return whether downstream use should expose a caution flag."""

        return self.decision == SourceDecision.ACCEPT_WITH_CAUTION

    @property
    def requires_review(self) -> bool:
        """Return whether a source should remain outside the active corpus."""

        return self.decision == SourceDecision.QUARANTINE
