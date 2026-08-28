from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from academic_literature_rag.services.generation_client import (
    GenerationResponse,
)


class OpenAIGenerationError(RuntimeError):
    """Raised when OpenAI text generation fails or returns unusable output."""


class OpenAIResponsesResource(Protocol):
    """Minimal protocol for the OpenAI Responses resource."""

    def create(
        self,
        *,
        model: str,
        input: str,
    ) -> object:
        """Create one OpenAI response."""


class OpenAIClient(Protocol):
    """Minimal protocol for the OpenAI SDK client."""

    @property
    def responses(
        self,
    ) -> OpenAIResponsesResource:
        """Return the Responses API resource."""


class OpenAIGenerationClient:
    """GenerationClient implementation backed by OpenAI Responses API."""

    def __init__(
        self,
        *,
        model_name: str = "gpt-5",
        client: OpenAIClient | None = None,
    ) -> None:
        cleaned_model_name = model_name.strip()

        if not cleaned_model_name:
            raise ValueError("OpenAI generation model name cannot be empty.")

        self._model_name = cleaned_model_name
        self._client = client or OpenAI()

    @property
    def model_name(
        self,
    ) -> str:
        """Return the OpenAI generation model name."""

        return self._model_name

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        """Generate one text answer from one prompt."""

        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise OpenAIGenerationError("Prompt cannot be empty.")

        try:
            response = self._client.responses.create(
                model=self._model_name,
                input=normalized_prompt,
            )
        except Exception as error:
            raise OpenAIGenerationError(
                f"OpenAI generation request failed: {type(error).__name__}: {error}"
            ) from error

        output_text = self._extract_output_text(response)

        if not output_text:
            raise OpenAIGenerationError("OpenAI returned an empty generation response.")

        return GenerationResponse(
            model=self._model_name,
            text=output_text,
        )

    @staticmethod
    def _extract_output_text(
        response: object,
    ) -> str:
        """Extract generated text from an OpenAI Responses API response."""

        output_text = getattr(
            response,
            "output_text",
            "",
        )

        if output_text is None:
            return ""

        if not isinstance(
            output_text,
            str,
        ):
            output_text = str(output_text)

        return output_text.strip()