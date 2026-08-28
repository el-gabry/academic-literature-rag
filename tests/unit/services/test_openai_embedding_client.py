from __future__ import annotations

from dataclasses import dataclass

import pytest

from academic_literature_rag.services.embedding_client import EmbeddingResponse
from academic_literature_rag.services.openai_embedding_client import (
    OpenAIEmbeddingClient,
    OpenAIEmbeddingError,
)


@dataclass
class FakeEmbeddingItem:
    embedding: object | None


@dataclass
class FakeEmbeddingResponse:
    data: object | None


class FakeEmbeddingsResource:
    def __init__(
        self,
        *,
        response: FakeEmbeddingResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    embedding=[
                        0.1,
                        0.2,
                        0.3,
                    ]
                )
            ]
        )
        self.error = error
        self.create_calls: list[dict[str, str]] = []

    def create(
        self,
        *,
        model: str,
        input: str,  # noqa: A002
    ) -> object:
        self.create_calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class FakeOpenAIClient:
    def __init__(
        self,
        embeddings: FakeEmbeddingsResource,
    ) -> None:
        self._embeddings = embeddings

    @property
    def embeddings(
        self,
    ) -> FakeEmbeddingsResource:
        return self._embeddings


def test_embed_text_calls_openai_embeddings_api_and_returns_response() -> None:
    embeddings = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    embedding=[
                        0.25,
                        0.5,
                        0.75,
                    ]
                )
            ]
        )
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    result = client.embed_text(
        "  retrieval augmented generation  "
    )

    assert result == EmbeddingResponse(
        model="text-embedding-3-small",
        vector=[
            0.25,
            0.5,
            0.75,
        ],
    )
    assert result.dimension == 3
    assert embeddings.create_calls == [
        {
            "model": "text-embedding-3-small",
            "input": "retrieval augmented generation",
        }
    ]


def test_model_name_is_stripped() -> None:
    embeddings = FakeEmbeddingsResource()
    client = OpenAIEmbeddingClient(
        model_name="  text-embedding-3-small  ",
        client=FakeOpenAIClient(embeddings),
    )

    assert client.model_name == "text-embedding-3-small"


def test_rejects_empty_model_name() -> None:
    embeddings = FakeEmbeddingsResource()

    with pytest.raises(
        ValueError,
        match="OpenAI embedding model name cannot be empty",
    ):
        OpenAIEmbeddingClient(
            model_name="   ",
            client=FakeOpenAIClient(embeddings),
        )


def test_rejects_empty_text() -> None:
    embeddings = FakeEmbeddingsResource()
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        OpenAIEmbeddingError,
        match="Text cannot be empty",
    ):
        client.embed_text(
            "   "
        )

    assert embeddings.create_calls == []


def test_wraps_openai_request_errors() -> None:
    embeddings = FakeEmbeddingsResource(
        error=RuntimeError("network failed")
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        OpenAIEmbeddingError,
        match="OpenAI embedding request failed: RuntimeError: network failed",
    ):
        client.embed_text(
            "retrieval augmented generation"
        )


def test_rejects_missing_data() -> None:
    embeddings = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            data=None
        )
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        OpenAIEmbeddingError,
        match="OpenAI returned an empty embedding vector",
    ):
        client.embed_text(
            "retrieval augmented generation"
        )


def test_rejects_empty_data() -> None:
    embeddings = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            data=[]
        )
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        OpenAIEmbeddingError,
        match="OpenAI returned an empty embedding vector",
    ):
        client.embed_text(
            "retrieval augmented generation"
        )


def test_rejects_missing_embedding_vector() -> None:
    embeddings = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    embedding=None
                )
            ]
        )
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        OpenAIEmbeddingError,
        match="OpenAI returned an empty embedding vector",
    ):
        client.embed_text(
            "retrieval augmented generation"
        )


def test_rejects_empty_embedding_vector() -> None:
    embeddings = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    embedding=[]
                )
            ]
        )
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    with pytest.raises(
        OpenAIEmbeddingError,
        match="OpenAI returned an empty embedding vector",
    ):
        client.embed_text(
            "retrieval augmented generation"
        )


def test_converts_embedding_values_to_floats() -> None:
    embeddings = FakeEmbeddingsResource(
        response=FakeEmbeddingResponse(
            data=[
                FakeEmbeddingItem(
                    embedding=[
                        1,
                        "2.5",
                        3.0,
                    ]
                )
            ]
        )
    )
    client = OpenAIEmbeddingClient(
        model_name="text-embedding-3-small",
        client=FakeOpenAIClient(embeddings),
    )

    result = client.embed_text(
        "retrieval augmented generation"
    )

    assert result == EmbeddingResponse(
        model="text-embedding-3-small",
        vector=[
            1.0,
            2.5,
            3.0,
        ],
    )