from __future__ import annotations

from typing import Any

import pytest
from openai import OpenAIError

from academic_literature_rag.services.openai_generation_client import (
    OpenAIGenerationClient,
    OpenAIGenerationClientError,
)


class FakeGenerationResponse:
    """Fake OpenAI generation response."""

    def __init__(
        self,
        *,
        output_text: str,
    ) -> None:
        self.output_text = output_text


class FakeResponsesResource:
    """Fake OpenAI responses resource."""

    def __init__(
        self,
        *,
        response: FakeGenerationResponse | None = None,
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
    ) -> FakeGenerationResponse:
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
    """Fake OpenAI client with a responses resource."""

    def __init__(
        self,
        responses: FakeResponsesResource,
    ) -> None:
        self.responses = responses


def test_generate_returns_generation_response() -> None:
    responses_resource = FakeResponsesResource(
        response=FakeGenerationResponse(
            output_text="Self-attention relates tokens to each other.",
        )
    )

    client = OpenAIGenerationClient(
        model_name="gpt-4.1-mini",
        client=FakeOpenAIClient(responses_resource),  # type: ignore[arg-type]
    )

    response = client.generate("Answer using the provided evidence.")

    assert response.model == "gpt-4.1-mini"
    assert response.text == "Self-attention relates tokens to each other."

    assert responses_resource.calls == [
        {
            "model": "gpt-4.1-mini",
            "input": "Answer using the provided evidence.",
        }
    ]


def test_generate_strips_prompt_before_request() -> None:
    responses_resource = FakeResponsesResource(
        response=FakeGenerationResponse(
            output_text="Grounded answer.",
        )
    )

    client = OpenAIGenerationClient(
        client=FakeOpenAIClient(responses_resource),  # type: ignore[arg-type]
    )

    client.generate("   Use evidence only.   ")

    assert responses_resource.calls == [
        {
            "model": "gpt-4.1-mini",
            "input": "Use evidence only.",
        }
    ]


def test_generate_rejects_empty_prompt() -> None:
    client = OpenAIGenerationClient(
        client=FakeOpenAIClient(
            FakeResponsesResource(
                response=FakeGenerationResponse(
                    output_text="Grounded answer.",
                )
            )
        ),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIGenerationClientError,
        match="prompt cannot be empty",
    ):
        client.generate("   ")


def test_init_rejects_empty_model_name() -> None:
    with pytest.raises(
        ValueError,
        match="model name cannot be empty",
    ):
        OpenAIGenerationClient(
            model_name="   ",
            client=FakeOpenAIClient(
                FakeResponsesResource(
                    response=FakeGenerationResponse(
                        output_text="Grounded answer.",
                    )
                )
            ),  # type: ignore[arg-type]
        )


def test_generate_wraps_openai_error() -> None:
    responses_resource = FakeResponsesResource(
        error=OpenAIError("API failure"),
    )

    client = OpenAIGenerationClient(
        client=FakeOpenAIClient(responses_resource),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIGenerationClientError,
        match="OpenAI generation request failed",
    ):
        client.generate("Use evidence only.")


def test_generate_rejects_empty_output_text() -> None:
    responses_resource = FakeResponsesResource(
        response=FakeGenerationResponse(
            output_text="   ",
        )
    )

    client = OpenAIGenerationClient(
        client=FakeOpenAIClient(responses_resource),  # type: ignore[arg-type]
    )

    with pytest.raises(
        OpenAIGenerationClientError,
        match="empty text",
    ):
        client.generate("Use evidence only.")