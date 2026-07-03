from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from academic_literature_rag.models.paper_candidate import PaperCandidate


def test_creates_valid_paper_candidate() -> None:
    paper = PaperCandidate(
        source="semantic_scholar",
        source_id="abc123",
        title="Continuous Glucose Monitoring in Older Adults",
        landing_url="https://example.com/paper",
        retrieved_at=datetime.now(UTC),
        authors=["Jane Smith", "John Doe"],
        publication_year=2025,
        doi="10.1000/example",
    )

    assert paper.source == "semantic_scholar"
    assert paper.title == "Continuous Glucose Monitoring in Older Adults"
    assert paper.authors == ["Jane Smith", "John Doe"]


def test_rejects_blank_title() -> None:
    with pytest.raises(ValidationError):
        PaperCandidate(
            source="arxiv",
            source_id="2501.12345",
            title="   ",
            landing_url="https://arxiv.org/abs/2501.12345",
            retrieved_at=datetime.now(UTC),
        )
