from __future__ import annotations

import pytest
from pydantic import ValidationError

from academic_literature_rag.models.research_intent import (
    ResearchIntent,
    ResearchNeed,
)


def build_research_intent(**overrides: object) -> ResearchIntent:
    """Build a valid research intent with optional field overrides."""

    values: dict[str, object] = {
        "research_idea": (
            "Investigate trustworthy change-point detection "
            "in physiological time series."
        ),
        "research_question": (
            "How can change-point detection be validated "
            "without reliable ground-truth labels?"
        ),
        "target_concepts": [
            "change-point detection",
            "reliability estimation",
        ],
        "primary_domains": [
            "physiological time series",
        ],
        "secondary_domains": [
            "financial time series",
        ],
        "research_needs": [
            ResearchNeed.METHODS,
            ResearchNeed.EMPIRICAL_EVIDENCE,
        ],
        "allow_cross_domain_transfer": True,
    }

    values.update(overrides)

    return ResearchIntent.model_validate(values)


def test_research_intent_normalizes_required_text() -> None:
    intent = build_research_intent(
        research_idea="  trustworthy   change-point   detection  ",
        research_question="  How   reliable   is   CPD?  ",
    )

    assert intent.research_idea == "trustworthy change-point detection"
    assert intent.research_question == "How reliable is CPD?"


def test_research_intent_normalizes_and_deduplicates_text_lists() -> None:
    intent = build_research_intent(
        target_concepts=[
            "  change-point detection ",
            "CHANGE-POINT DETECTION",
            " reliability ",
            "",
        ],
        secondary_domains=[
            " Finance ",
            "finance",
            " industrial monitoring ",
        ],
    )

    assert intent.target_concepts == [
        "change-point detection",
        "reliability",
    ]

    assert intent.secondary_domains == [
        "Finance",
        "industrial monitoring",
    ]


def test_research_intent_deduplicates_research_needs() -> None:
    intent = build_research_intent(
        research_needs=[
            ResearchNeed.METHODS,
            ResearchNeed.METHODS,
            ResearchNeed.BENCHMARKS,
        ]
    )

    assert intent.research_needs == [
        ResearchNeed.METHODS,
        ResearchNeed.BENCHMARKS,
    ]


def test_research_intent_allows_cross_domain_transfer() -> None:
    intent = build_research_intent(
        allow_cross_domain_transfer=True,
        secondary_domains=[
            "financial time series",
            "industrial monitoring",
        ],
    )

    assert intent.allow_cross_domain_transfer is True
    assert intent.secondary_domains == [
        "financial time series",
        "industrial monitoring",
    ]


def test_research_intent_can_disable_cross_domain_transfer() -> None:
    intent = build_research_intent(
        allow_cross_domain_transfer=False,
    )

    assert intent.allow_cross_domain_transfer is False


def test_research_intent_rejects_blank_research_question() -> None:
    with pytest.raises(ValidationError):
        build_research_intent(
            research_question="   ",
        )


def test_research_intent_requires_target_concepts() -> None:
    with pytest.raises(ValidationError):
        build_research_intent(
            target_concepts=[],
        )


def test_research_intent_requires_primary_domain() -> None:
    with pytest.raises(ValidationError):
        build_research_intent(
            primary_domains=[],
        )


def test_research_intent_requires_research_need() -> None:
    with pytest.raises(ValidationError):
        build_research_intent(
            research_needs=[],
        )