from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from academic_literature_rag.identity.normalizers import (
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)
from academic_literature_rag.models.paper_candidate import PaperCandidate


class PaperMatchKind(StrEnum):
    """Possible outcomes when comparing two source-paper records."""

    EXACT_DOI = "exact_doi"
    EXACT_ARXIV_ID = "exact_arxiv_id"
    TITLE_CANDIDATE = "title_candidate"
    CONFLICT = "conflict"
    NO_MATCH = "no_match"


@dataclass(frozen=True)
class PaperIdentity:
    """Normalized identifiers used to compare source-paper records."""

    doi: str | None
    arxiv_id: str | None
    normalized_title: str | None

    @classmethod
    def from_candidate(
        cls,
        candidate: PaperCandidate,
    ) -> PaperIdentity:
        """Build a comparable identity from a source-paper candidate."""

        return cls(
            doi=normalize_doi(candidate.doi),
            arxiv_id=normalize_arxiv_id(candidate.arxiv_id),
            normalized_title=normalize_title(candidate.title),
        )


@dataclass(frozen=True)
class PaperMatchDecision:
    """Explain whether two records can be treated as the same paper."""

    kind: PaperMatchKind
    reason: str

    @property
    def can_auto_merge(self) -> bool:
        """Return whether the decision is safe for automatic merging."""

        return self.kind in {
            PaperMatchKind.EXACT_DOI,
            PaperMatchKind.EXACT_ARXIV_ID,
        }


def compare_paper_candidates(
    left: PaperCandidate,
    right: PaperCandidate,
) -> PaperMatchDecision:
    """Compare two source-paper candidates conservatively."""

    return compare_paper_identities(
        PaperIdentity.from_candidate(left),
        PaperIdentity.from_candidate(right),
    )


def compare_paper_identities(
    left: PaperIdentity,
    right: PaperIdentity,
) -> PaperMatchDecision:
    """Compare two normalized identities using conservative rules."""

    doi_matches = left.doi is not None and right.doi is not None and left.doi == right.doi

    arxiv_id_matches = (
        left.arxiv_id is not None and right.arxiv_id is not None and left.arxiv_id == right.arxiv_id
    )

    doi_conflicts = left.doi is not None and right.doi is not None and left.doi != right.doi

    arxiv_id_conflicts = (
        left.arxiv_id is not None and right.arxiv_id is not None and left.arxiv_id != right.arxiv_id
    )

    if doi_matches and arxiv_id_conflicts:
        return PaperMatchDecision(
            kind=PaperMatchKind.CONFLICT,
            reason=("The records share a DOI but contain different arXiv identifiers."),
        )

    if arxiv_id_matches and doi_conflicts:
        return PaperMatchDecision(
            kind=PaperMatchKind.CONFLICT,
            reason=("The records share an arXiv identifier but contain different DOIs."),
        )

    if doi_matches:
        return PaperMatchDecision(
            kind=PaperMatchKind.EXACT_DOI,
            reason="The records share the same normalized DOI.",
        )

    if arxiv_id_matches:
        return PaperMatchDecision(
            kind=PaperMatchKind.EXACT_ARXIV_ID,
            reason="The records share the same normalized arXiv identifier.",
        )

    if (
        left.normalized_title is not None
        and right.normalized_title is not None
        and left.normalized_title == right.normalized_title
    ):
        return PaperMatchDecision(
            kind=PaperMatchKind.TITLE_CANDIDATE,
            reason=(
                "The records have identical normalized titles, but no shared strong identifier."
            ),
        )

    return PaperMatchDecision(
        kind=PaperMatchKind.NO_MATCH,
        reason="The records do not share a reliable identity signal.",
    )
