from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from academic_literature_rag.connectors.arxiv import (
    ArxivClient,
    ArxivRequestError,
    ArxivResponseError,
)


FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "arxiv_search.xml"


def load_fixture() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def test_search_maps_atom_xml_to_paper_candidates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/query"
        assert request.url.params["search_query"] == ('all:"continuous glucose monitoring"')
        assert request.url.params["max_results"] == "2"

        return httpx.Response(
            status_code=200,
            text=load_fixture(),
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = ArxivClient(http_client=http_client)

        papers = client.search(
            query="continuous glucose monitoring",
            limit=2,
        )

    assert len(papers) == 2

    first_paper = papers[0]

    assert first_paper.source == "arxiv"
    assert first_paper.source_id == "2501.12345v2"
    assert first_paper.arxiv_id == "2501.12345"
    assert first_paper.title == "Continuous Glucose Monitoring in Older Adults"
    assert first_paper.authors == ["Jane Smith", "John Doe"]
    assert first_paper.publication_year == 2025
    assert first_paper.doi == "10.1000/arxiv.example.001"
    assert first_paper.open_access_pdf_url == ("http://arxiv.org/pdf/2501.12345v2")

    second_paper = papers[1]

    assert second_paper.source_id == "2502.67890v1"
    assert second_paper.arxiv_id == "2502.67890"
    assert second_paper.doi is None
    assert second_paper.open_access_pdf_url is None
    assert second_paper.citation_count is None


def test_search_rejects_blank_query() -> None:
    with httpx.Client() as http_client:
        client = ArxivClient(http_client=http_client)

        with pytest.raises(ValueError, match="must not be blank"):
            client.search(query="   ")


def test_invalid_xml_raises_response_error() -> None:
    with httpx.Client() as http_client:
        client = ArxivClient(http_client=http_client)

        with pytest.raises(ArxivResponseError, match="invalid Atom XML"):
            client.map_search_feed("<feed><entry>")


def test_http_failure_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=503,
            text="Service unavailable",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = ArxivClient(http_client=http_client)

        with pytest.raises(ArxivRequestError):
            client.search(query="continuous glucose monitoring")
