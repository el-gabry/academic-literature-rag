from __future__ import annotations

import re

from academic_literature_rag.models.research_intent import (
    ResearchIntent,
    ResearchNeed,
)


class ResearchIntentConsistencyService:
    """Conservatively preserve explicit research needs from the original idea.

    The service does not reinterpret the research question, concepts, or
    domains. It only restores research-need categories that are strongly
    supported by explicit wording in the researcher's original input.
    """

    _NEED_PATTERNS: dict[ResearchNeed, tuple[str, ...]] = {
        ResearchNeed.METHODS: (
            r"\bmethod\b",
            r"\bmethods\b",
            r"\bmethodology\b",
            r"\bmethodologies\b",
            r"\btechnique\b",
            r"\btechniques\b",
            r"\balgorithm\b",
            r"\balgorithms\b",
        ),
        ResearchNeed.EMPIRICAL_EVIDENCE: (
            r"\bempirical evidence\b",
            r"\bempirical evaluation\b",
            r"\bexperimental validation\b",
            r"\bexperimental evaluation\b",
            r"\breal[- ]world validation\b",
            r"\breal[- ]data validation\b",
            r"\bvalidated\b",
            r"\bvalidation\b",
            r"\bevaluated\b",
            r"\bevaluation\b",
        ),
        ResearchNeed.SYNTHESIS: (
            r"\breview literature\b",
            r"\bliterature review\b",
            r"\bsystematic review\b",
            r"\bscoping review\b",
            r"\breview paper\b",
            r"\breview papers\b",
            r"\bsurvey paper\b",
            r"\bsurvey papers\b",
            r"\bliterature overview\b",
            r"\bstate[- ]of[- ]the[- ]art\b",
        ),
        ResearchNeed.FOUNDATIONAL_CONTEXT: (
            r"\bfoundational work\b",
            r"\bfoundational literature\b",
            r"\btheoretical background\b",
            r"\btheoretical foundation\b",
            r"\btheoretical foundations\b",
            r"\bdefinitions\b",
        ),
        ResearchNeed.BENCHMARKS: (
            r"\bbenchmark\b",
            r"\bbenchmarks\b",
            r"\bbenchmark dataset\b",
            r"\bbenchmark datasets\b",
            r"\bbenchmark study\b",
            r"\bbenchmark studies\b",
        ),
        ResearchNeed.LIMITATIONS: (
            r"\blimitation\b",
            r"\blimitations\b",
            r"\bweakness\b",
            r"\bweaknesses\b",
            r"\bfailure condition\b",
            r"\bfailure conditions\b",
        ),
        ResearchNeed.COUNTER_EVIDENCE: (
            r"\bcounter[- ]evidence\b",
            r"\bcontradictory evidence\b",
            r"\bconflicting evidence\b",
            r"\bopposing evidence\b",
            r"\bnegative evidence\b",
            r"\bcontradictory findings\b",
            r"\bconflicting findings\b",
        ),
    }

    def ensure_consistency(
        self,
        intent: ResearchIntent,
    ) -> ResearchIntent:
        """Restore explicit research needs omitted during LLM parsing."""

        detected_needs = self._detect_explicit_needs(
            intent.research_idea,
        )

        merged_needs = list(intent.research_needs)

        for research_need in detected_needs:
            if research_need not in merged_needs:
                merged_needs.append(research_need)

        if merged_needs == intent.research_needs:
            return intent

        return intent.model_copy(
            update={
                "research_needs": merged_needs,
            }
        )

    def _detect_explicit_needs(
        self,
        research_idea: str,
    ) -> list[ResearchNeed]:
        """Return research needs explicitly signaled in the original idea."""

        normalized_idea = " ".join(
            research_idea.casefold().split()
        )

        detected_needs: list[ResearchNeed] = []

        for research_need, patterns in self._NEED_PATTERNS.items():
            if any(
                re.search(pattern, normalized_idea)
                for pattern in patterns
            ):
                detected_needs.append(research_need)

        return detected_needs