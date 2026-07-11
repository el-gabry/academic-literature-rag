from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationResponse:
    """Text generation output returned by an LLM provider."""

    model: str
    text: str


class GenerationClient(Protocol):
    """Interface for LLM text generation providers."""

    @property
    def model_name(
        self,
    ) -> str:
        """Return the generation model name."""

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        """Generate one answer from one prompt."""
