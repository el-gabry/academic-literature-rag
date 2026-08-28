from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from academic_literature_rag.services.embedding_client import (
    EmbeddingResponse,
)


class OpenAIEmbeddingError(RuntimeError):
    """Raised when OpenAI embedding generation fails or returns unusable output."""


class OpenAIEmbeddingsResource(Protocol):
    """Minimal protocol for the OpenAI Embeddings resource."""

    def create(
        self,
        *,
        model: str,
        input: str,
    ) -> object:
        """Create one OpenAI embedding response."""


class OpenAIClient(Protocol):
    """Minimal protocol for the OpenAI SDK client."""

    @property
    def embeddings(
        self,
    ) -> OpenAIEmbeddingsResource:
        """Return the Embeddings API resource."""


class OpenAIEmbeddingClient:
    """EmbeddingClient implementation backed by OpenAI Embeddings API."""

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-3-small",
        client: OpenAIClient | None = None,
    ) -> None:
        cleaned_model_name = model_name.strip()

        if not cleaned_model_name:
            raise ValueError("OpenAI embedding model name cannot be empty.")

        self._model_name = cleaned_model_name
        self._client = client or OpenAI()

    @property
    def model_name(
        self,
    ) -> str:
        """Return the OpenAI embedding model name."""

        return self._model_name

    def embed_text(
        self,
        text: str,
    ) -> EmbeddingResponse:
        """Generate one embedding vector from one text input."""

        normalized_text = text.strip()

        if not normalized_text:
            raise OpenAIEmbeddingError("Text cannot be empty.")

        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=normalized_text,
            )
        except Exception as error:
            raise OpenAIEmbeddingError(
                f"OpenAI embedding request failed: {type(error).__name__}: {error}"
            ) from error

        vector = self._extract_embedding_vector(response)

        if not vector:
            raise OpenAIEmbeddingError("OpenAI returned an empty embedding vector.")

        return EmbeddingResponse(
            model=self._model_name,
            vector=vector,
        )

    @staticmethod
    def _extract_embedding_vector(
        response: object,
    ) -> list[float]:
        """Extract the first embedding vector from an OpenAI embedding response."""

        data = getattr(
            response,
            "data",
            None,
        )

        if not data:
            return []

        first_item = data[0]
        embedding = getattr(
            first_item,
            "embedding",
            None,
        )

        if embedding is None:
            return []

        return [float(value) for value in embedding]