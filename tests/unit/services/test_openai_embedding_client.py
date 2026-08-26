from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from openai import OpenAIError

from academic_literature_rag.services.openai_embedding_client import (
    OpenAIEmbeddingClient,
    OpenAIEmbeddingClientError,
)


@dataclass(frozen=True)
class FakeEmbeddingData:
    """Fake embedding item returned by OpenAI."""

    embedding: list[float]


@dataclass(frozen=True)
class FakeEmbeddingResponse:
    """Fake OpenAI embedding response."""

    model: str
    data: list[FakeEmbeddingData]


class FakeEmbeddingsResource:
    """Fake OpenAI embeddings resource."""

    def __init__(
        self,
        *,
        response: FakeEmbeddingResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        input: str,
    ) -> FakeEmbeddingResponse:
        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        if self._error is not None:
            raise self._error

        assert self._response is not None

        return self._response


class FakeOpenAIClient:
    """Fake OpenAI client with an embeddings resource."""

    def __init__(
        self,
        embeddings: FakeEmbeddingsResource,
    ) -> None:
        self.embeddings = embeddings


def test_embed_text_generates_embedding_response() -> None:
    embeddings_resource = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            model="text-embedding-3-small",
            data=[
                FakeEmbeddingData(
                    embedding=[
                        0.1,
                        0.2,
                        0.3,
                    ],
                )
            ],
        )
    )

    fake_client = FakeOpenAIClient(embeddings_resource)

    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=fake_client,  # type: ignore[arg-type]
    )

    response = client.embed_text("What is self-attention?")

    assert response.model == "text-embedding-3-small"
    assert response.vector == [
        0.1,
        0.2,
        0.3,
    ]
    assert response.dimension == 3

    assert embeddings_resource.calls == [
        {
            "model": "text-embedding-3-small",
            "input": "What is self-attention?",
        }
    ]


def test_embed_text_strips_input_before_request() -> None:
    embeddings_resource = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            model="text-embedding-3-small",
            data=[
                FakeEmbeddingData(
                    embedding=[
                        1.0,
                        2.0,
                    ],
                )
            ],
        )
    )

    client = OpenAIEmbeddingClient(
        client=FakeOpenAIClient(embeddings_resource),  # type: ignore[arg-type]
    )

    client.embed_text("   Transformer attention   ")

    assert embeddings_resource.calls == [
        {
            "model": "text-embedding-3-small",
            "input": "Transformer attention",
        }
    ]


def test_embed_text_rejects_empty_input() -> None:
    client = OpenAIEmbeddingClient(
        client=FakeOpenAIClient(
            FakeEmbeddingsResource(
                response=FakeEmbeddingResponse(
                    model="text-embedding-3-small",
                    data=[
                        FakeEmbeddingData(
                            embedding=[1.0],
                        )
                    ],
                )
            )
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIEmbeddingClientError,
        match="cannot be empty",
    ):
        client.embed_text("   ")


def test_init_rejects_empty_model_name() -> None:
    with pytest.raises(
        ValueError,
        match="model name cannot be empty",
    ):
        OpenAIEmbeddingClient(
            model_name="   ",
            client=FakeOpenAIClient(
                FakeEmbeddingsResource(
                    response=FakeEmbeddingResponse(
                        model="text-embedding-3-small",
                        data=[
                            FakeEmbeddingData(
                                embedding=[1.0],
                            )
                        ],
                    )
                )
            ),  # type: ignore[arg-type]
        )


def test_embed_text_wraps_openai_error() -> None:
    embeddings_resource = FakeEmbeddingsResource(
        error=OpenAIError("API failure"),
    )

    client = OpenAIEmbeddingClient(
        client=FakeOpenAIClient(embeddings_resource),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIEmbeddingClientError,
        match="OpenAI embedding request failed",
    ):
        client.embed_text("attention")


def test_embed_text_rejects_response_without_data() -> None:
    embeddings_resource = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            model="text-embedding-3-small",
            data=[],
        )
    )

    client = OpenAIEmbeddingClient(
        client=FakeOpenAIClient(embeddings_resource),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIEmbeddingClientError,
        match="did not contain data",
    ):
        client.embed_text("attention")


def test_embed_text_rejects_empty_embedding_vector() -> None:
    embeddings_resource = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            model="text-embedding-3-small",
            data=[
                FakeEmbeddingData(
                    embedding=[],
                )
            ],
        )
    )

    client = OpenAIEmbeddingClient(
        client=FakeOpenAIClient(embeddings_resource),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIEmbeddingClientError,
        match="empty vector",
    ):
        client.embed_text("attention")