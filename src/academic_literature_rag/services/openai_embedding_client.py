from __future__ import annotations

from openai import OpenAI, OpenAIError

from academic_literature_rag.services.embedding_client import (
    EmbeddingResponse,
)


class OpenAIEmbeddingClientError(RuntimeError):
    """Raised when OpenAI embedding generation fails."""


class OpenAIEmbeddingClient:
    """OpenAI implementation of the embedding client interface."""

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-3-small",
        client: OpenAI | None = None,
    ) -> None:
        self._model_name = self._normalize_model_name(model_name)
        self._client = client or OpenAI()

    @property
    def model_name(
        self,
    ) -> str:
        """Return the embedding model name."""

        return self._model_name

    def embed_text(
        self,
        text: str,
    ) -> EmbeddingResponse:
        """Generate one OpenAI embedding vector for one text input."""

        normalized_text = text.strip()

        if not normalized_text:
            raise OpenAIEmbeddingClientError(
                "Embedding input text cannot be empty."
            )

        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=normalized_text,
            )
        except OpenAIError as error:
            raise OpenAIEmbeddingClientError(
                f"OpenAI embedding request failed: {error}"
            ) from error

        if not response.data:
            raise OpenAIEmbeddingClientError(
                "OpenAI embedding response did not contain data."
            )

        embedding_vector = response.data[0].embedding

        if not embedding_vector:
            raise OpenAIEmbeddingClientError(
                "OpenAI embedding response contained an empty vector."
            )

        return EmbeddingResponse(
            model=response.model,
            vector=[
                float(value)
                for value in embedding_vector
            ],
        )

    @staticmethod
    def _normalize_model_name(
        model_name: str,
    ) -> str:
        normalized_model_name = model_name.strip()

        if not normalized_model_name:
            raise ValueError("OpenAI embedding model name cannot be empty.")

        return normalized_model_name