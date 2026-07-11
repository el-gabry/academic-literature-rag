from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class AnswerCitation(BaseModel):
    """Citation linking an answer to a retrieved text chunk."""

    text_chunk_id: UUID
    pdf_asset_id: UUID

    chunk_index: int

    start_page_number: int
    end_page_number: int

    similarity_score: float
    cited_text: str

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(
        cls,
        value: int,
    ) -> int:
        """Reject invalid chunk indexes."""

        if value < 0:
            raise ValueError("Citation chunk index must be zero or greater.")

        return value

    @field_validator("start_page_number", "end_page_number")
    @classmethod
    def validate_page_number(
        cls,
        value: int,
    ) -> int:
        """Reject invalid page numbers."""

        if value < 1:
            raise ValueError("Citation page number must be at least 1.")

        return value

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(
        cls,
        value: float,
    ) -> float:
        """Reject impossible cosine similarity scores."""

        if value < -1.0 or value > 1.0:
            raise ValueError("Citation similarity score must be between -1 and 1.")

        return value

    @field_validator("cited_text")
    @classmethod
    def validate_cited_text(
        cls,
        value: str,
    ) -> str:
        """Reject empty citation text."""

        normalized_text = value.strip()

        if not normalized_text:
            raise ValueError("Citation text cannot be empty.")

        return normalized_text

    @model_validator(mode="after")
    def validate_page_range(
        self,
    ) -> AnswerCitation:
        """Reject invalid citation page ranges."""

        if self.end_page_number < self.start_page_number:
            raise ValueError("Citation end page cannot be before start page.")

        return self


class GroundedAnswer(BaseModel):
    """Answer generated from retrieved evidence chunks."""

    question: str
    answer: str

    citations: list[AnswerCitation]

    generation_model: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        """Reject empty questions."""

        normalized_question = value.strip()

        if not normalized_question:
            raise ValueError("Grounded answer question cannot be empty.")

        return normalized_question

    @field_validator("answer")
    @classmethod
    def validate_answer(
        cls,
        value: str,
    ) -> str:
        """Reject empty answers."""

        normalized_answer = value.strip()

        if not normalized_answer:
            raise ValueError("Grounded answer text cannot be empty.")

        return normalized_answer

    @field_validator("citations")
    @classmethod
    def validate_citations(
        cls,
        value: list[AnswerCitation],
    ) -> list[AnswerCitation]:
        """Require at least one citation for grounded answers."""

        if not value:
            raise ValueError("Grounded answer must contain at least one citation.")

        return value

    @field_validator("generation_model")
    @classmethod
    def validate_generation_model(
        cls,
        value: str,
    ) -> str:
        """Reject empty generation model names."""

        normalized_value = value.strip()

        if not normalized_value:
            raise ValueError("Generation model cannot be empty.")

        return normalized_value
