from __future__ import annotations

from dataclasses import dataclass

import pytest

from academic_literature_rag.services.generation_client import GenerationResponse
from academic_literature_rag.services.openai_generation_client import (
    OpenAIGenerationClient,
    OpenAIGenerationError,
)


@dataclass
class FakeOpenAIResponse:
    output_text: object | None


class FakeResponsesResource:
    def __init__(
        self,
        *,
        response: FakeOpenAIResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or FakeOpenAIResponse(
            output_text="Generated answer from OpenAI."
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
        responses: FakeResponsesResource,
    ) -> None:
        self._responses = responses

    @property
    def responses(
        self,
    ) -> FakeResponsesResource:
        return self._responses


def test_generate_calls_openai_responses_api_and_returns_generation_response() -> None:
    responses = FakeResponsesResource(
        response=FakeOpenAIResponse(
            output_text="  Grounded generated answer.  "
        )
    )
    client = OpenAIGenerationClient(
        model_name="gpt-5",
        client=FakeOpenAIClient(responses),
    )

    result = client.generate(
        "  Use the evidence only.  "
    )

    assert result == GenerationResponse(
        model="gpt-5",
        text="Grounded generated answer.",
    )
    assert responses.create_calls == [
        {
            "model": "gpt-5",
            "input": "Use the evidence only.",
        }
    ]


def test_model_name_is_stripped() -> None:
    responses = FakeResponsesResource()
    client = OpenAIGenerationClient(
        model_name="  gpt-5  ",
        client=FakeOpenAIClient(responses),
    )

    assert client.model_name == "gpt-5"


def test_rejects_empty_model_name() -> None:
    responses = FakeResponsesResource()

    with pytest.raises(
        ValueError,
        match="OpenAI generation model name cannot be empty",
    ):
        OpenAIGenerationClient(
            model_name="   ",
            client=FakeOpenAIClient(responses),
        )


def test_rejects_empty_prompt() -> None:
    responses = FakeResponsesResource()
    client = OpenAIGenerationClient(
        model_name="gpt-5",
        client=FakeOpenAIClient(responses),
    )

    with pytest.raises(
        OpenAIGenerationError,
        match="Prompt cannot be empty",
    ):
        client.generate(
            "   "
        )

    assert responses.create_calls == []


def test_wraps_openai_request_errors() -> None:
    responses = FakeResponsesResource(
        error=RuntimeError("network failed")
    )
    client = OpenAIGenerationClient(
        model_name="gpt-5",
        client=FakeOpenAIClient(responses),
    )

    with pytest.raises(
        OpenAIGenerationError,
        match="OpenAI generation request failed: RuntimeError: network failed",
    ):
        client.generate(
            "Answer this question."
        )


def test_rejects_missing_output_text() -> None:
    responses = FakeResponsesResource(
        response=FakeOpenAIResponse(
            output_text=None
        )
    )
    client = OpenAIGenerationClient(
        model_name="gpt-5",
        client=FakeOpenAIClient(responses),
    )

    with pytest.raises(
        OpenAIGenerationError,
        match="OpenAI returned an empty generation response",
    ):
        client.generate(
            "Answer this question."
        )


def test_rejects_blank_output_text() -> None:
    responses = FakeResponsesResource(
        response=FakeOpenAIResponse(
            output_text="   "
        )
    )
    client = OpenAIGenerationClient(
        model_name="gpt-5",
        client=FakeOpenAIClient(responses),
    )

    with pytest.raises(
        OpenAIGenerationError,
        match="OpenAI returned an empty generation response",
    ):
        client.generate(
            "Answer this question."
        )


def test_converts_non_string_output_text_to_string() -> None:
    responses = FakeResponsesResource(
        response=FakeOpenAIResponse(
            output_text=123
        )
    )
    client = OpenAIGenerationClient(
        model_name="gpt-5",
        client=FakeOpenAIClient(responses),
    )

    result = client.generate(
        "Answer this question."
    )

    assert result == GenerationResponse(
        model="gpt-5",
        text="123",
    )