from __future__ import annotations

from pathlib import Path

import pytest

from academic_literature_rag.app.factory import (
    RagCoreServices,
    RagRepositories,
    RagServiceFactory,
)
from academic_literature_rag.config import (
    AppConfig,
    ConfigError,
    DemoConfig,
    OpenAIConfig,
    StorageConfig,
)
from academic_literature_rag.services.rag_answer_service import RagAnswerService
from academic_literature_rag.services.rag_pipeline_service import RagPipelineService


def make_config(
    tmp_path: Path,
    *,
    api_key: str | None = "sk-test",
) -> AppConfig:
    return AppConfig(
        openai=OpenAIConfig(
            api_key=api_key,
            generation_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        ),
        storage=StorageConfig(
            database_path=tmp_path / "db" / "dev.db",
            raw_response_dir=tmp_path / "raw_responses",
            pdf_dir=tmp_path / "pdfs",
        ),
        demo=DemoConfig(
            search_query="transformer attention",
            question="What is attention in transformers?",
            retrieval_limit=2,
            download_limit=1,
            embedding_limit=3,
            top_k=3,
        ),
        semantic_scholar_api_key=None,
    )


def test_from_config_creates_storage_directories_and_repositories(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    factory = RagServiceFactory.from_config(config)

    try:
        assert config.storage.database_path.parent.exists()
        assert config.storage.raw_response_dir.exists()
        assert config.storage.pdf_dir.exists()

        assert isinstance(
            factory.repositories,
            RagRepositories,
        )
        assert factory.config is config
    finally:
        factory.close()


def test_from_config_rejects_missing_openai_api_key(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        api_key=None,
    )

    with pytest.raises(
        ConfigError,
        match="OPENAI_API_KEY is not set",
    ):
        RagServiceFactory.from_config(config)


def test_create_core_services_returns_service_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "sk-test",
    )

    factory = RagServiceFactory.from_config(
        make_config(tmp_path)
    )

    try:
        services = factory.create_core_services()

        assert isinstance(
            services,
            RagCoreServices,
        )
        assert isinstance(
            services.rag_answer_service,
            RagAnswerService,
        )
    finally:
        factory.close()


def test_create_pipeline_service_returns_pipeline_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "sk-test",
    )

    factory = RagServiceFactory.from_config(
        make_config(tmp_path)
    )

    try:
        pipeline_service = factory.create_pipeline_service()

        assert isinstance(
            pipeline_service,
            RagPipelineService,
        )
    finally:
        factory.close()


def test_create_answer_service_returns_answer_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "sk-test",
    )

    factory = RagServiceFactory.from_config(
        make_config(tmp_path)
    )

    try:
        answer_service = factory.create_answer_service()

        assert isinstance(
            answer_service,
            RagAnswerService,
        )
    finally:
        factory.close()


def test_close_can_be_called_more_than_once(
    tmp_path: Path,
) -> None:
    factory = RagServiceFactory.from_config(
        make_config(tmp_path)
    )

    factory.close()
    factory.close()