from __future__ import annotations

from academic_literature_rag.models.paper_candidate import PaperCandidate
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


class SourceAssessmentPolicy:
    """Conservatively assess whether a retrieved paper should enter the corpus.

    This first policy evaluates source provenance and metadata quality only.
    It does not claim that a paper's scientific conclusions are trustworthy.

    Later versions can enrich this assessment using PubMed, Crossref,
    retraction databases, publication-type metadata, and topic classifiers.
    """

    def assess(
        self,
        candidate: PaperCandidate,
        *,
        topic_relevance: TopicRelevance = TopicRelevance.UNKNOWN,
    ) -> SourceAssessment:
        """Return an explainable source-intake assessment."""

        authority = self._authority(candidate)
        publication_status = self._publication_status(candidate)
        peer_review_status = self._peer_review_status(candidate)
        metadata_confidence = self._metadata_confidence(candidate)

        decision, reasons = self._decision(
            candidate=candidate,
            authority=authority,
            metadata_confidence=metadata_confidence,
            topic_relevance=topic_relevance,
        )

        return SourceAssessment(
            provider=candidate.source,
            authority=authority,
            publication_status=publication_status,
            peer_review_status=peer_review_status,
            evidence_type=EvidenceType.UNKNOWN,
            retraction_status=RetractionStatus.UNKNOWN,
            metadata_confidence=metadata_confidence,
            topic_relevance=topic_relevance,
            decision=decision,
            reasons=reasons,
        )

    @staticmethod
    def _authority(candidate: PaperCandidate) -> SourceAuthority:
        if candidate.source == "arxiv":
            return SourceAuthority.PREPRINT_REPOSITORY

        if candidate.source == "semantic_scholar":
            return SourceAuthority.AGGREGATOR

        return SourceAuthority.UNKNOWN

    @staticmethod
    def _publication_status(candidate: PaperCandidate) -> PublicationStatus:
        if candidate.source == "arxiv":
            return PublicationStatus.PREPRINT

        # Semantic Scholar aggregates documents with different publication
        # states. Do not infer peer-reviewed publication merely from presence
        # in the index.
        return PublicationStatus.UNKNOWN

    @staticmethod
    def _peer_review_status(candidate: PaperCandidate) -> PeerReviewStatus:
        if candidate.source == "arxiv":
            return PeerReviewStatus.UNVERIFIED

        # The current PaperCandidate model does not contain authoritative
        # peer-review metadata.
        return PeerReviewStatus.UNKNOWN

    @staticmethod
    def _metadata_confidence(
        candidate: PaperCandidate,
    ) -> AssessmentConfidence:
        has_strong_identifier = bool(candidate.doi or candidate.arxiv_id)
        has_authors = bool(candidate.authors)
        has_year = candidate.publication_year is not None
        has_venue = bool(candidate.venue)

        completeness_signals = sum(
            (
                has_strong_identifier,
                has_authors,
                has_year,
                has_venue,
            )
        )

        if has_strong_identifier and completeness_signals >= 3:
            return AssessmentConfidence.HIGH

        if has_strong_identifier or completeness_signals >= 2:
            return AssessmentConfidence.MEDIUM

        return AssessmentConfidence.LOW

    def _decision(
        self,
        *,
        candidate: PaperCandidate,
        authority: SourceAuthority,
        metadata_confidence: AssessmentConfidence,
        topic_relevance: TopicRelevance,
    ) -> tuple[SourceDecision, tuple[str, ...]]:
        reasons: list[str] = []

        if topic_relevance == TopicRelevance.IRRELEVANT:
            return (
                SourceDecision.REJECT,
                ("The paper was assessed as irrelevant to the target topic.",),
            )

        if metadata_confidence == AssessmentConfidence.LOW:
            reasons.append(
                "The paper has insufficient metadata for confident automatic intake."
            )

            if authority == SourceAuthority.UNKNOWN:
                reasons.append("The source authority is unknown.")

            return SourceDecision.QUARANTINE, tuple(reasons)

        if candidate.source == "arxiv":
            reasons.append(
                "The paper comes from a recognized scientific preprint repository."
            )
            reasons.append(
                "Peer-review status is not verified by the current metadata."
            )

            if topic_relevance == TopicRelevance.LOW:
                reasons.append("Topic relevance is currently low.")

            return SourceDecision.ACCEPT_WITH_CAUTION, tuple(reasons)

        if candidate.source == "semantic_scholar":
            reasons.append(
                "The paper was discovered through a recognized scholarly index."
            )

            if candidate.doi:
                reasons.append("A DOI is available as a strong identity signal.")

            if candidate.arxiv_id:
                reasons.append(
                    "An arXiv identifier is available as a strong identity signal."
                )

            if topic_relevance == TopicRelevance.LOW:
                reasons.append(
                    "The paper has low topic relevance and requires manual review."
                )
                return SourceDecision.QUARANTINE, tuple(reasons)

            if metadata_confidence == AssessmentConfidence.HIGH:
                reasons.append(
                    "Metadata completeness is sufficient for automatic corpus intake."
                )
                return SourceDecision.ACCEPT, tuple(reasons)

            reasons.append(
                "Metadata is usable, but additional source verification is desirable."
            )
            return SourceDecision.ACCEPT_WITH_CAUTION, tuple(reasons)

        return (
            SourceDecision.QUARANTINE,
            ("No intake policy exists yet for this source provider.",),
        )