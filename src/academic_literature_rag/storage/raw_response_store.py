from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID


class RawResponseStore:
    """Stores original API responses without modifying their content."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def save_json(
        self,
        *,
        source: str,
        run_id: UUID,
        payload: Mapping[str, Any],
    ) -> Path:
        cleaned_source = self._validate_source(source)

        target_path = self._base_dir / cleaned_source / f"{run_id}.json"

        if target_path.exists():
            raise FileExistsError(f"Raw response already exists: {target_path}")

        target_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = target_path.with_name(f"{target_path.name}.tmp")

        try:
            temporary_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            temporary_path.replace(target_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        return target_path

    @staticmethod
    def _validate_source(source: str) -> str:
        cleaned_source = source.strip()

        if not cleaned_source:
            raise ValueError("Source must not be blank.")

        if cleaned_source in {".", ".."}:
            raise ValueError("Source must be a safe directory name.")

        if "/" in cleaned_source or "\\" in cleaned_source:
            raise ValueError("Source must not contain path separators.")

        return cleaned_source
