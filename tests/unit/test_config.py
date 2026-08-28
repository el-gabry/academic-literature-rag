from __future__ import annotations

from pathlib import Path

import pytest

from academic_literature_rag.config import (
    AppConfig,
    ConfigError,
    OpenAIConfig,
    StorageConfig,
)


def test_from_env_uses_defaults_when_env_is_empty() -> None:
    config = AppConfig.from_env({})

    assert config.openai.api_key is None
    assert config.openai.generation_model == "gpt-4o-mini"
    assert config.openai.embedding_model == "text-embedding-3-small"

    assert config.semantic_scholar_api_key is None

    assert config.demo.search_query == "transformer attention"
    assert config.demo.question == "What is attention in transformers?"
    assert config.demo.retrieval_limit == 2
    assert config.demo.download_limit == 1
    assert config.demo.embedding_limit == 8
    assert config.demo.top_k == 3

    assert config.storage.database_path == Path("data/db/dev.db")
    assert config.storage.raw_response_dir == Path("data/raw_responses")
    assert config.storage.pdf_dir == Path("data/pdfs")


def test_from_env_reads_custom_values() -> None:
    config = AppConfig.from_env(
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_GENERATION_MODEL": "gpt-4o-mini",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
            "SEMANTIC_SCHOLAR_API_KEY": "semantic-test",
            "RAG_DEMO_SEARCH_QUERY": "medical image retrieval",
            "RAG_DEMO_QUESTION": "How is retrieval used in medical AI?",
            "RAG_DEMO_RETRIEVAL_LIMIT": "5",
            "RAG_DEMO_DOWNLOAD_LIMIT": "2",
            "RAG_DEMO_EMBEDDING_LIMIT": "20",
            "RAG_DEMO_TOP_K": "7",
            "RAG_DEMO_DATABASE_PATH": "custom/db.sqlite",
            "RAG_DEMO_RAW_RESPONSE_DIR": "custom/raw",
            "RAG_DEMO_PDF_DIR": "custom/pdfs",
        }
    )

    assert config.openai.api_key == "sk-test"
    assert config.openai.generation_model == "gpt-4o-mini"
    assert config.openai.embedding_model == "text-embedding-3-large"
    assert config.semantic_scholar_api_key == "semantic-test"

    assert config.demo.search_query == "medical image retrieval"
    assert config.demo.question == "How is retrieval used in medical AI?"
    assert config.demo.retrieval_limit == 5
    assert config.demo.download_limit == 2
    assert config.demo.embedding_limit == 20
    assert config.demo.top_k == 7

    assert config.storage.database_path == Path("custom/db.sqlite")
    assert config.storage.raw_response_dir == Path("custom/raw")
    assert config.storage.pdf_dir == Path("custom/pdfs")


def test_optional_env_values_are_stripped() -> None:
    config = AppConfig.from_env(
        {
            "OPENAI_API_KEY": "  sk-test  ",
            "SEMANTIC_SCHOLAR_API_KEY": "  semantic-test  ",
        }
    )

    assert config.openai.api_key == "sk-test"
    assert config.semantic_scholar_api_key == "semantic-test"


def test_blank_optional_env_values_become_none() -> None:
    config = AppConfig.from_env(
        {
            "OPENAI_API_KEY": "   ",
            "SEMANTIC_SCHOLAR_API_KEY": "   ",
        }
    )

    assert config.openai.api_key is None
    assert config.semantic_scholar_api_key is None


def test_openai_config_require_api_key_returns_key() -> None:
    config = OpenAIConfig(
        api_key="sk-test",
        generation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    )

    assert config.require_api_key() == "sk-test"


def test_openai_config_require_api_key_raises_clear_error() -> None:
    config = OpenAIConfig(
        api_key=None,
        generation_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    )

    with pytest.raises(
        ConfigError,
        match="OPENAI_API_KEY is not set",
    ):
        config.require_api_key()


def test_storage_config_create_directories(
    tmp_path: Path,
) -> None:
    storage = StorageConfig(
        database_path=tmp_path / "db" / "dev.db",
        raw_response_dir=tmp_path / "raw_responses",
        pdf_dir=tmp_path / "pdfs",
    )

    storage.create_directories()

    assert storage.database_path.parent.exists()
    assert storage.raw_response_dir.exists()
    assert storage.pdf_dir.exists()


@pytest.mark.parametrize(
    "env_name",
    [
        "OPENAI_GENERATION_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "RAG_DEMO_SEARCH_QUERY",
        "RAG_DEMO_QUESTION",
        "RAG_DEMO_DATABASE_PATH",
        "RAG_DEMO_RAW_RESPONSE_DIR",
        "RAG_DEMO_PDF_DIR",
    ],
)
def test_from_env_rejects_empty_required_text_values(
    env_name: str,
) -> None:
    with pytest.raises(
        ConfigError,
        match=f"{env_name} cannot be empty",
    ):
        AppConfig.from_env(
            {
                env_name: "   ",
            }
        )


@pytest.mark.parametrize(
    "env_name",
    [
        "RAG_DEMO_RETRIEVAL_LIMIT",
        "RAG_DEMO_DOWNLOAD_LIMIT",
        "RAG_DEMO_EMBEDDING_LIMIT",
        "RAG_DEMO_TOP_K",
    ],
)
def test_from_env_rejects_non_integer_limits(
    env_name: str,
) -> None:
    with pytest.raises(
        ConfigError,
        match=f"{env_name} must be an integer",
    ):
        AppConfig.from_env(
            {
                env_name: "abc",
            }
        )


@pytest.mark.parametrize(
    "env_name",
    [
        "RAG_DEMO_RETRIEVAL_LIMIT",
        "RAG_DEMO_DOWNLOAD_LIMIT",
        "RAG_DEMO_EMBEDDING_LIMIT",
        "RAG_DEMO_TOP_K",
    ],
)
def test_from_env_rejects_limits_below_one(
    env_name: str,
) -> None:
    with pytest.raises(
        ConfigError,
        match=f"{env_name} must be at least 1",
    ):
        AppConfig.from_env(
            {
                env_name: "0",
            }
        )