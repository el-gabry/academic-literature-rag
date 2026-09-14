from __future__ import annotations

from dataclasses import dataclass

import pytest

from academic_literature_rag.models.research_intent import ResearchNeed
from academic_literature_rag.services.generation_client import (
    GenerationResponse,
)
from academic_literature_rag.services.llm_research_intent_parser import (
    LlmResearchIntentParser,
)
from academic_literature_rag.services.research_intent_parser import (
    ResearchIntentParsingError,
)


@dataclass
class FakeGenerationClient:
    """Simple fake generation client for parser unit tests."""

    response_text: str
    model_name: str = "fake-model"

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        return GenerationResponse(
            model=self.model_name,
            text=self.response_text,
        )


class FailingGenerationClient:
    """Generation client that always raises an error."""

    model_name = "fake-model"

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        raise RuntimeError("generation failed")


def test_parser_returns_valid_research_intent() -> None:
    client = FakeGenerationClient(
        response_text="""
        {
          "research_question": "How can CPD be validated without labels?",
          "target_concepts": [
            "change-point detection",
            "reliability estimation"
          ],
          "primary_domains": [
            "physiological time series"
          ],
          "secondary_domains": [],
          "research_needs": [
            "methods",
            "empirical_evidence"
          ],
          "allow_cross_domain_transfer": false
        }
        """
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    intent = parser.parse(
        "Study trustworthy change-point detection in physiological signals."
    )

    assert intent.research_question == (
        "How can CPD be validated without labels?"
    )

    assert intent.target_concepts == [
        "change-point detection",
        "reliability estimation",
    ]

    assert intent.primary_domains == [
        "physiological time series",
    ]

    assert intent.research_needs == [
        ResearchNeed.METHODS,
        ResearchNeed.EMPIRICAL_EVIDENCE,
    ]

    assert intent.allow_cross_domain_transfer is False


def test_parser_preserves_cross_domain_request_without_inventing_domain() -> None:
    client = FakeGenerationClient(
        response_text="""
        {
          "research_question": "Which validation methods transfer across domains?",
          "target_concepts": [
            "change-point detection",
            "validation"
          ],
          "primary_domains": [
            "physiological time series"
          ],
          "secondary_domains": [],
          "research_needs": [
            "methods"
          ],
          "allow_cross_domain_transfer": true
        }
        """
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    intent = parser.parse(
        "Study CPD validation in physiological data and include methods "
        "from other domains."
    )

    assert intent.allow_cross_domain_transfer is True
    assert intent.secondary_domains == []


def test_parser_preserves_explicit_secondary_domain() -> None:
    client = FakeGenerationClient(
        response_text="""
        {
          "research_question": "Can financial CPD methods transfer to health?",
          "target_concepts": [
            "change-point detection",
            "transferability"
          ],
          "primary_domains": [
            "physiological time series"
          ],
          "secondary_domains": [
            "financial time series"
          ],
          "research_needs": [
            "methods"
          ],
          "allow_cross_domain_transfer": true
        }
        """
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    intent = parser.parse(
        "Study whether CPD methods from finance can transfer to "
        "physiological time series."
    )

    assert intent.secondary_domains == [
        "financial time series",
    ]


def test_parser_normalizes_duplicate_concepts() -> None:
    client = FakeGenerationClient(
        response_text="""
        {
          "research_question": "How reliable is CPD?",
          "target_concepts": [
            "CPD",
            "cpd",
            " reliability "
          ],
          "primary_domains": [
            "time series"
          ],
          "secondary_domains": [],
          "research_needs": [
            "methods"
          ],
          "allow_cross_domain_transfer": false
        }
        """
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    intent = parser.parse(
        "Study CPD reliability."
    )

    assert intent.target_concepts == [
        "CPD",
        "reliability",
    ]


def test_parser_rejects_invalid_json() -> None:
    client = FakeGenerationClient(
        response_text="not valid json"
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    with pytest.raises(
        ResearchIntentParsingError,
        match="invalid JSON",
    ):
        parser.parse(
            "Study trustworthy CPD."
        )


def test_parser_rejects_invalid_research_need() -> None:
    client = FakeGenerationClient(
        response_text="""
        {
          "research_question": "How reliable is CPD?",
          "target_concepts": [
            "change-point detection"
          ],
          "primary_domains": [
            "time series"
          ],
          "secondary_domains": [],
          "research_needs": [
            "magic_evidence"
          ],
          "allow_cross_domain_transfer": false
        }
        """
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    with pytest.raises(
        ResearchIntentParsingError,
        match="does not match",
    ):
        parser.parse(
            "Study trustworthy CPD."
        )


def test_parser_rejects_blank_research_idea() -> None:
    client = FakeGenerationClient(
        response_text="{}"
    )

    parser = LlmResearchIntentParser(
        generation_client=client,
    )

    with pytest.raises(
        ResearchIntentParsingError,
        match="cannot be blank",
    ):
        parser.parse(
            "   "
        )


def test_parser_wraps_generation_failure() -> None:
    parser = LlmResearchIntentParser(
        generation_client=FailingGenerationClient(),
    )

    with pytest.raises(
        ResearchIntentParsingError,
        match="generation failed",
    ):
        parser.parse(
            "Study trustworthy CPD."
        )