from __future__ import annotations

import json
from pathlib import Path

import httpx

from academic_literature_rag.connectors.semantic_scholar import (
    SemanticScholarClient,
)
from academic_literature_rag.services.retrieval_service import (
    SemanticScholarRetrievalService,
)
from academic_literature_rag.storage.raw_response_store import (
    RawResponseStore,
)


FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "semantic_scholar_search.json"


def load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_search_saves_raw_response_before_normalizing(tmp_path) -> None:
    fixture_payload = load_fixture()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=fixture_payload,
            request=request,
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as http_client:
        client = SemanticScholarClient(http_client=http_client)

        store = RawResponseStore(base_dir=tmp_path / "raw")

        service = SemanticScholarRetrievalService(
            client=client,
            raw_response_store=store,
        )

        result = service.search(
            query="continuous glucose monitoring",
            limit=2,
        )

    assert result.run.status == "completed"
    assert result.run.result_count == 2
    assert result.run.raw_response_path is not None
    assert len(result.papers) == 2

    raw_response_path = Path(result.run.raw_response_path)

    assert raw_response_path.exists()

    saved_payload = json.loads(raw_response_path.read_text(encoding="utf-8"))

    assert saved_payload == fixture_payload
