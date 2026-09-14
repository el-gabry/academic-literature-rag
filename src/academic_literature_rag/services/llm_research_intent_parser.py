from __future__ import annotations

import json

from pydantic import ValidationError

from academic_literature_rag.models.research_intent import (
    ResearchIntent,
    ResearchNeed,
)
from academic_literature_rag.services.generation_client import GenerationClient
from academic_literature_rag.services.research_intent_parser import (
    ResearchIntentParsingError,
)


class LlmResearchIntentParser:
    """Parse free-text research ideas into validated ResearchIntent objects."""

    def __init__(
        self,
        *,
        generation_client: GenerationClient,
    ) -> None:
        self._generation_client = generation_client

    def parse(
        self,
        research_idea: str,
    ) -> ResearchIntent:
        """Convert one free-text research idea into a structured intent."""

        normalized_idea = " ".join(research_idea.split())

        if not normalized_idea:
            raise ResearchIntentParsingError(
                "Research idea cannot be blank."
            )

        prompt = self._build_prompt(normalized_idea)

        try:
            response = self._generation_client.generate(prompt)
        except Exception as error:
            raise ResearchIntentParsingError(
                "Research intent generation failed."
            ) from error

        raw_output = response.text.strip()

        if not raw_output:
            raise ResearchIntentParsingError(
                "Research intent parser returned empty output."
            )

        return self._parse_output(
            raw_output=raw_output,
            original_research_idea=normalized_idea,
        )

    @staticmethod
    def _build_prompt(
        research_idea: str,
    ) -> str:
        """Build the instruction used to extract scientific research intent."""

        allowed_research_needs = ", ".join(
            research_need.value
            for research_need in ResearchNeed
        )

        return (
            "You are extracting structured scientific research intent.\n"
            "Analyze the researcher's idea conservatively and faithfully.\n"
            "Do not invent scientific domains, methods, evidence needs, or "
            "research goals that are not supported by the input.\n\n"
            "Return ONLY one valid JSON object with exactly these fields:\n"
            "{\n"
            '  "research_question": "string",\n'
            '  "target_concepts": ["string"],\n'
            '  "primary_domains": ["string"],\n'
            '  "secondary_domains": ["string"],\n'
            '  "research_needs": ["string"],\n'
            '  "allow_cross_domain_transfer": true\n'
            "}\n\n"
            "Definitions:\n"
            "- target_concepts describe WHAT scientific methods, phenomena, "
            "properties, or constructs are being investigated.\n"
            "- primary_domains describe WHERE or in what scientific/data "
            "context the research is being conducted.\n"
            "- secondary_domains contain only explicitly named additional "
            "domains from which evidence may be transferred.\n"
            "- research_needs describe WHAT TYPES of scientific evidence the "
            "researcher wants to find.\n\n"
            "Rules for research_question:\n"
            "- Formulate one clear scientific question.\n"
            "- Preserve the researcher's actual objective.\n"
            "- Do not make the question broader than the supplied idea.\n\n"
            "Rules for target_concepts:\n"
            "- Include scientific methods, constructs, properties, or "
            "phenomena central to the question.\n"
            "- Examples include change-point detection, uncertainty, "
            "reliability, robustness, calibration, or validation when those "
            "concepts are actually present in the idea.\n"
            "- Do not put datasets, populations, application areas, or "
            "scientific domains into target_concepts.\n\n"
            "Rules for primary_domains:\n"
            "- Use the most specific scientific or data domain supported by "
            "the research idea.\n"
            "- Prefer a specific domain such as 'physiological time series' "
            "over broad umbrella labels such as 'data science', "
            "'computer science', or 'physiology'.\n"
            "- Do not invent a more specific application than the researcher "
            "provided.\n\n"
            "Rules for secondary_domains:\n"
            "- Include only domains explicitly named by the researcher.\n"
            "- Do not invent finance, climate, manufacturing, medicine, or "
            "other example domains.\n"
            "- If the researcher requests transferable or cross-domain "
            "evidence without naming a specific secondary domain, return an "
            "empty secondary_domains list.\n\n"
            "Rules for cross-domain transfer:\n"
            "- Set allow_cross_domain_transfer to true only when the "
            "researcher explicitly requests, permits, or clearly seeks "
            "transferable evidence from other domains.\n"
            "- Otherwise set it to false.\n\n"
            "Rules for research_needs:\n"
            "- Allowed values are: "
            f"{allowed_research_needs}.\n"
            "- Use 'methods' for requests about methods, techniques, "
            "approaches, algorithms, or methodology.\n"
            "- Use 'empirical_evidence' for experimental validation, "
            "evaluations, observations, real-data studies, or evidence about "
            "whether a method works in practice.\n"
            "- Use 'synthesis' for surveys, reviews, literature overviews, "
            "systematic reviews, or state-of-the-art summaries.\n"
            "- Use 'foundational_context' for theoretical background, "
            "definitions, principles, or foundational literature.\n"
            "- Use 'benchmarks' for benchmark datasets, benchmark studies, "
            "or comparative evaluations.\n"
            "- Use 'limitations' for weaknesses, failure conditions, "
            "robustness issues, sensitivity, or known limitations.\n"
             "- Use 'counter_evidence' for contradictory, negative, "
            "conflicting, opposing, or challenging scientific evidence.\n"
            "- If the researcher explicitly asks for contradictory, "
            "conflicting, opposing, or negative evidence, include "
            "'counter_evidence' even when 'limitations' is also included.\n"
            "- Do not treat limitations and counter-evidence as synonyms: "
            "limitations describe weaknesses or boundary conditions, while "
            "counter-evidence describes scientific findings that challenge "
            "or conflict with a claim, method, or conclusion.\n"
            "- Include every research need clearly supported by the input, "
            "but do not add categories merely for completeness.\n\n"
            "Output requirements:\n"
            "- Return JSON only.\n"
            "- Do not include markdown fences.\n"
            "- Do not include explanations before or after the JSON.\n\n"
            "Research idea:\n"
            f"{research_idea}"
        )

    @staticmethod
    def _parse_output(
        *,
        raw_output: str,
        original_research_idea: str,
    ) -> ResearchIntent:
        """Validate model JSON and convert it into the domain model."""

        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as error:
            raise ResearchIntentParsingError(
                "Research intent parser returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise ResearchIntentParsingError(
                "Research intent parser output must be a JSON object."
            )

        payload["research_idea"] = original_research_idea

        try:
            return ResearchIntent.model_validate(payload)
        except ValidationError as error:
            raise ResearchIntentParsingError(
                "Research intent parser output does not match "
                "the ResearchIntent schema."
            ) from error