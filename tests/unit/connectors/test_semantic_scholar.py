from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarRequestError,
)


FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "semantic_scholar_search.json"


def load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_search_maps_api_response_to_paper_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/graph/v1/paper/search"
        assert request.url.params["query"] == "continuous glucose monitoring"
        assert request.url.params["limit"] == "2"

        return httpx.Response(
            status_code=200,
            json=load_fixture(),
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = SemanticScholarClient(http_client=http_client)

        papers = client.search(
            query="continuous glucose monitoring",
            limit=2,
        )

    assert len(papers) == 2

    first_paper = papers[0]

    assert first_paper.source == "semantic_scholar"
    assert first_paper.source_id == "paper-001"
    assert first_paper.title == "Continuous Glucose Monitoring in Older Adults"
    assert first_paper.authors == ["Jane Smith", "John Doe"]
    assert first_paper.doi == "10.1000/example.001"
    assert first_paper.open_access_pdf_url == "https://example.org/paper-001.pdf"
    assert first_paper.citation_count == 12

    second_paper = papers[1]

    assert second_paper.abstract is None
    assert second_paper.authors == []
    assert second_paper.doi is None
    assert second_paper.open_access_pdf_url is None


def test_search_rejects_blank_query() -> None:
    with httpx.Client() as http_client:
        client = SemanticScholarClient(http_client=http_client)

        with pytest.raises(ValueError, match="must not be blank"):
            client.search("   ")


def test_search_wraps_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            json={"message": "Too many requests"},
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = SemanticScholarClient(http_client=http_client)

        with pytest.raises(SemanticScholarRequestError):
            client.search("continuous glucose monitoring")
