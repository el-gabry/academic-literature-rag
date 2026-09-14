from __future__ import annotations

from academic_literature_rag.models.research_intent import (
    ResearchIntent,
    ResearchNeed,
)
from academic_literature_rag.services.research_intent_consistency_service import (
    ResearchIntentConsistencyService,
)


def build_intent(
    *,
    research_idea: str,
    research_needs: list[ResearchNeed],
) -> ResearchIntent:
    """Build a valid research intent for consistency tests."""

    return ResearchIntent(
        research_idea=research_idea,
        research_question="How can change-point detection be validated?",
        target_concepts=[
            "change-point detection",
            "reliability",
        ],
        primary_domains=[
            "physiological time series",
        ],
        secondary_domains=[],
        research_needs=research_needs,
        allow_cross_domain_transfer=False,
    )


def test_adds_empirical_evidence_when_validation_is_explicit() -> None:
    intent = build_intent(
        research_idea=(
            "I want experimental validation of change-point detection "
            "in physiological time series."
        ),
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.METHODS,
        ResearchNeed.EMPIRICAL_EVIDENCE,
    ]


def test_adds_synthesis_when_review_literature_is_explicit() -> None:
    intent = build_intent(
        research_idea=(
            "I want to study change-point detection and review literature "
            "on existing approaches."
        ),
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.METHODS,
        ResearchNeed.SYNTHESIS,
    ]


def test_adds_counter_evidence_when_contradiction_is_explicit() -> None:
    intent = build_intent(
        research_idea=(
            "I want to understand contradictory evidence about "
            "change-point detection reliability."
        ),
        research_needs=[
            ResearchNeed.LIMITATIONS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.LIMITATIONS,
        ResearchNeed.COUNTER_EVIDENCE,
    ]


def test_keeps_limitations_and_counter_evidence_separate() -> None:
    intent = build_intent(
        research_idea=(
            "I want to study methods, limitations, and contradictory evidence "
            "for change-point detection."
        ),
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.METHODS,
        ResearchNeed.LIMITATIONS,
        ResearchNeed.COUNTER_EVIDENCE,
    ]


def test_adds_multiple_explicit_needs() -> None:
    intent = build_intent(
        research_idea=(
            "I want experimental validation, benchmark studies, "
            "review literature, limitations, and contradictory evidence "
            "for change-point detection."
        ),
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.METHODS,
        ResearchNeed.EMPIRICAL_EVIDENCE,
        ResearchNeed.SYNTHESIS,
        ResearchNeed.BENCHMARKS,
        ResearchNeed.LIMITATIONS,
        ResearchNeed.COUNTER_EVIDENCE,
    ]


def test_does_not_invent_unrequested_needs() -> None:
    intent = build_intent(
        research_idea=(
            "I want to study change-point detection reliability "
            "in physiological time series."
        ),
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.METHODS,
    ]


def test_does_not_duplicate_existing_needs() -> None:
    intent = build_intent(
        research_idea=(
            "I want benchmark studies and review literature "
            "for change-point detection."
        ),
        research_needs=[
            ResearchNeed.BENCHMARKS,
            ResearchNeed.SYNTHESIS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result.research_needs == [
        ResearchNeed.BENCHMARKS,
        ResearchNeed.SYNTHESIS,
    ]


def test_returns_same_object_when_no_repair_is_needed() -> None:
    intent = build_intent(
        research_idea=(
            "I want methods for change-point detection "
            "in physiological time series."
        ),
        research_needs=[
            ResearchNeed.METHODS,
        ],
    )

    service = ResearchIntentConsistencyService()

    result = service.ensure_consistency(intent)

    assert result is intent