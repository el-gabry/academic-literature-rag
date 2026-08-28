from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import NoReturn, cast

from academic_literature_rag.app.demo_output import (
    SUPPORTED_DEMO_OUTPUT_FORMATS,
    DemoOutputFormat,
    render_demo_output,
)
from academic_literature_rag.app.factory import RagServiceFactory
from academic_literature_rag.app.logging_config import (
    SUPPORTED_LOG_FORMATS,
    LogFormat,
    LoggingConfigError,
    configure_logging,
)
from academic_literature_rag.config import AppConfig, ConfigError
from academic_literature_rag.repositories.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)

logger = logging.getLogger(__name__)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the Academic Literature RAG command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "demo":
            return run_demo_command(
                output_format=cast(
                    DemoOutputFormat,
                    args.output_format,
                ),
                log_level=args.log_level,
                log_format=cast(
                    LogFormat,
                    args.log_format,
                ),
            )

        parser.print_help()
        return 0

    except (ConfigError, LoggingConfigError) as error:
        print_error(str(error))
        return 2

    except Exception as error:
        print_error(f"{type(error).__name__}: {error}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="academic-literature-rag",
        description=(
            "Academic Literature RAG: retrieve papers, process PDFs, "
            "embed chunks, search evidence, and generate grounded answers."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the full OpenAI-backed end-to-end RAG demo.",
    )
    demo_parser.add_argument(
        "--format",
        dest="output_format",
        choices=SUPPORTED_DEMO_OUTPUT_FORMATS,
        default="text",
        help="Output format for the demo result.",
    )
    demo_parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        help="Logging level, for example DEBUG, INFO, WARNING, or ERROR.",
    )
    demo_parser.add_argument(
        "--log-format",
        dest="log_format",
        choices=SUPPORTED_LOG_FORMATS,
        default="text",
        help="Logging format.",
    )
    demo_parser.set_defaults(
        command="demo",
    )

    return parser


def run_demo_command(
    *,
    output_format: DemoOutputFormat,
    log_level: str,
    log_format: LogFormat,
) -> int:
    """Run the full OpenAI-backed RAG demo."""

    configure_logging(
        log_level=log_level,
        log_format=log_format,
    )

    logger.info("Loading application configuration.")
    config = AppConfig.from_env()

    logger.info("Creating application services.")
    factory = RagServiceFactory.from_config(config)

    try:
        pipeline_service = factory.create_pipeline_service()

        logger.info(
            "Starting RAG ingestion for query: %s",
            config.demo.search_query,
        )
        ingestion_result = pipeline_service.ingest(
            query=config.demo.search_query,
            retrieval_limit=config.demo.retrieval_limit,
            download_limit=config.demo.download_limit,
            embedding_limit=config.demo.embedding_limit,
        )

        logger.info("Checking embedded chunks.")
        ensure_embeddings_exist(
            chunk_embedding_repository=factory.repositories.chunk_embedding_repository,
            embedding_model=config.openai.embedding_model,
        )

        logger.info(
            "Generating grounded answer for question: %s",
            config.demo.question,
        )
        answer_service = factory.create_answer_service()

        answer = answer_service.answer(
            config.demo.question,
            top_k=config.demo.top_k,
        )

    finally:
        logger.info("Closing application services.")
        factory.close()

    logger.info("Rendering demo output as %s.", output_format)
    output = render_demo_output(
        search_query=config.demo.search_query,
        question=config.demo.question,
        embedding_model=config.openai.embedding_model,
        generation_model=config.openai.generation_model,
        ingestion_result=ingestion_result,
        answer=answer,
        output_format=output_format,
    )

    print(output)

    logger.info("Demo completed successfully.")
    return 0


def ensure_embeddings_exist(
    *,
    chunk_embedding_repository: ChunkEmbeddingRepository,
    embedding_model: str,
) -> None:
    """Fail clearly if no embeddings exist after ingestion."""

    embeddings = chunk_embedding_repository.list_by_model(embedding_model)

    if embeddings:
        logger.info(
            "Found %s embeddings for model %s.",
            len(embeddings),
            embedding_model,
        )
        return

    raise RuntimeError(
        f"No embeddings found for model '{embedding_model}' after ingestion. "
        "Try increasing RAG_DEMO_RETRIEVAL_LIMIT, RAG_DEMO_DOWNLOAD_LIMIT, "
        "or RAG_DEMO_EMBEDDING_LIMIT."
    )


def print_error(
    message: str,
) -> None:
    """Print a CLI error message."""

    print(
        f"Error: {message}",
        file=sys.stderr,
    )


def exit_with_main() -> NoReturn:
    """Console script wrapper."""

    raise SystemExit(main())


if __name__ == "__main__":
    exit_with_main()