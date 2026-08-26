from __future__ import annotations

from openai import OpenAI, OpenAIError

from academic_literature_rag.services.generation_client import (
    GenerationResponse,
)


class OpenAIGenerationClientError(RuntimeError):
    """Raised when OpenAI text generation fails."""


class OpenAIGenerationClient:
    """OpenAI implementation of the generation client interface."""

    def __init__(
        self,
        *,
        model_name: str = "gpt-4.1-mini",
        client: OpenAI | None = None,
    ) -> None:
        self._model_name = self._normalize_model_name(model_name)
        self._client = client or OpenAI()

    @property
    def model_name(
        self,
    ) -> str:
        """Return the generation model name."""

        return self._model_name

    def generate(
        self,
        prompt: str,
    ) -> GenerationResponse:
        """Generate one grounded answer from one prompt."""

        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise OpenAIGenerationClientError(
                "Generation prompt cannot be empty."
            )

        try:
            response = self._client.responses.create(
                model=self._model_name,
                input=normalized_prompt,
            )
        except OpenAIError as error:
            raise OpenAIGenerationClientError(
                f"OpenAI generation request failed: {error}"
            ) from error

        output_text = getattr(
            response,
            "output_text",
            "",
        ).strip()

        if not output_text:
            raise OpenAIGenerationClientError(
                "OpenAI generation response contained empty text."
            )

        return GenerationResponse(
            model=self._model_name,
            text=output_text,
        )

    @staticmethod
    def _normalize_model_name(
        model_name: str,
    ) -> str:
        normalized_model_name = model_name.strip()

        if not normalized_model_name:
            raise ValueError("OpenAI generation model name cannot be empty.")

        return normalized_model_name