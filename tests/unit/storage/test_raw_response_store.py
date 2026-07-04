import json
from uuid import UUID

import pytest

from academic_literature_rag.storage.raw_response_store import (
    RawResponseStore,
)


def test_saves_json_response_in_source_directory(tmp_path) -> None:
    store = RawResponseStore(base_dir=tmp_path / "raw")

    run_id = UUID("12345678-1234-5678-1234-567812345678")
    payload = {
        "total": 1,
        "data": [{"paperId": "paper-001", "title": "Example Paper"}],
    }

    saved_path = store.save_json(
        source="semantic_scholar",
        run_id=run_id,
        payload=payload,
    )

    assert saved_path.exists()
    assert saved_path.name == f"{run_id}.json"
    assert saved_path.parent.name == "semantic_scholar"

    saved_payload = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved_payload == payload


def test_refuses_to_overwrite_existing_raw_response(tmp_path) -> None:
    store = RawResponseStore(base_dir=tmp_path / "raw")
    run_id = UUID("12345678-1234-5678-1234-567812345678")

    store.save_json(
        source="semantic_scholar",
        run_id=run_id,
        payload={"data": []},
    )

    with pytest.raises(FileExistsError):
        store.save_json(
            source="semantic_scholar",
            run_id=run_id,
            payload={"data": []},
        )


def test_rejects_unsafe_source_name(tmp_path) -> None:
    store = RawResponseStore(base_dir=tmp_path / "raw")

    with pytest.raises(ValueError, match="path separators"):
        store.save_json(
            source="../unsafe",
            run_id=UUID("12345678-1234-5678-1234-567812345678"),
            payload={"data": []},
        )
