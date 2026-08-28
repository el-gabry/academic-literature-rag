from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when application configuration is invalid."""


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI provider configuration."""

    api_key: str | None
    generation_model: str
    embedding_model: str

    def require_api_key(
        self,
    ) -> str:
        """Return the API key or raise a clear configuration error."""

        if self.api_key:
            return self.api_key

        raise ConfigError(
            "OPENAI_API_KEY is not set. "
            "Set it in your shell or local .env before running OpenAI scripts."
        )


@dataclass(frozen=True)
class StorageConfig:
    """Local storage paths for the RAG pipeline."""

    database_path: Path
    raw_response_dir: Path
    pdf_dir: Path

    def create_directories(
        self,
    ) -> None:
        """Create required local storage directories."""

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.raw_response_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.pdf_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


@dataclass(frozen=True)
class DemoConfig:
    """End-to-end demo configuration."""

    search_query: str
    question: str
    retrieval_limit: int
    download_limit: int
    embedding_limit: int
    top_k: int


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    openai: OpenAIConfig
    storage: StorageConfig
    demo: DemoConfig
    semantic_scholar_api_key: str | None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> AppConfig:
        """Build application configuration from environment variables."""

        resolved_env = os.environ if env is None else env

        return cls(
            openai=OpenAIConfig(
                api_key=_optional_env(
                    resolved_env,
                    "OPENAI_API_KEY",
                ),
                generation_model=_required_text_env(
                    resolved_env,
                    "OPENAI_GENERATION_MODEL",
                    default="gpt-4o-mini",
                ),
                embedding_model=_required_text_env(
                    resolved_env,
                    "OPENAI_EMBEDDING_MODEL",
                    default="text-embedding-3-small",
                ),
            ),
            storage=StorageConfig(
                database_path=Path(
                    _required_text_env(
                        resolved_env,
                        "RAG_DEMO_DATABASE_PATH",
                        default="data/db/dev.db",
                    )
                ),
                raw_response_dir=Path(
                    _required_text_env(
                        resolved_env,
                        "RAG_DEMO_RAW_RESPONSE_DIR",
                        default="data/raw_responses",
                    )
                ),
                pdf_dir=Path(
                    _required_text_env(
                        resolved_env,
                        "RAG_DEMO_PDF_DIR",
                        default="data/pdfs",
                    )
                ),
            ),
            demo=DemoConfig(
                search_query=_required_text_env(
                    resolved_env,
                    "RAG_DEMO_SEARCH_QUERY",
                    default="transformer attention",
                ),
                question=_required_text_env(
                    resolved_env,
                    "RAG_DEMO_QUESTION",
                    default="What is attention in transformers?",
                ),
                retrieval_limit=_positive_int_env(
                    resolved_env,
                    "RAG_DEMO_RETRIEVAL_LIMIT",
                    default=2,
                ),
                download_limit=_positive_int_env(
                    resolved_env,
                    "RAG_DEMO_DOWNLOAD_LIMIT",
                    default=1,
                ),
                embedding_limit=_positive_int_env(
                    resolved_env,
                    "RAG_DEMO_EMBEDDING_LIMIT",
                    default=8,
                ),
                top_k=_positive_int_env(
                    resolved_env,
                    "RAG_DEMO_TOP_K",
                    default=3,
                ),
            ),
            semantic_scholar_api_key=_optional_env(
                resolved_env,
                "SEMANTIC_SCHOLAR_API_KEY",
            ),
        )


def _optional_env(
    env: Mapping[str, str],
    name: str,
) -> str | None:
    """Return a stripped optional environment value."""

    value = env.get(name)

    if value is None:
        return None

    cleaned_value = value.strip()

    return cleaned_value or None


def _required_text_env(
    env: Mapping[str, str],
    name: str,
    *,
    default: str,
) -> str:
    """Return a required text environment value with a safe default."""

    value = env.get(
        name,
        default,
    ).strip()

    if not value:
        raise ConfigError(f"{name} cannot be empty.")

    return value


def _positive_int_env(
    env: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    """Return a positive integer environment value."""

    raw_value = env.get(
        name,
        str(default),
    ).strip()

    if not raw_value:
        raise ConfigError(f"{name} cannot be empty.")

    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigError(f"{name} must be an integer.") from error

    if value < 1:
        raise ConfigError(f"{name} must be at least 1.")

    return value