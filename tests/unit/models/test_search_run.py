from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from academic_literature_rag.models.search_run import SearchRun


def test_creates_search_run() -> None:
    run = SearchRun(
        source="semantic_scholar",
        query="continuous glucose monitoring",
        started_at=datetime(2026, 7, 4, tzinfo=UTC),
    )

    assert run.source == "semantic_scholar"
    assert run.status == "running"
    assert run.result_count is None
    assert run.query == "continuous glucose monitoring"


def test_rejects_blank_query() -> None:
    with pytest.raises(ValidationError, match="must not be blank"):
        SearchRun(
            source="arxiv",
            query="   ",
        )


def test_rejects_negative_result_count() -> None:
    with pytest.raises(ValidationError):
        SearchRun(
            source="semantic_scholar",
            query="glucose",
            result_count=-1,
        )
