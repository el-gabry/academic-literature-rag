from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest

from academic_literature_rag.app.logging_config import (
    JsonLogFormatter,
    LoggingConfigError,
    build_formatter,
    configure_logging,
    parse_log_level,
)


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Restore root logger state after each test."""

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    yield

    root_logger.handlers.clear()
    root_logger.handlers.extend(original_handlers)
    root_logger.setLevel(original_level)


def test_parse_log_level_accepts_standard_level_names() -> None:
    assert parse_log_level("DEBUG") == logging.DEBUG
    assert parse_log_level("INFO") == logging.INFO
    assert parse_log_level("WARNING") == logging.WARNING
    assert parse_log_level("ERROR") == logging.ERROR


def test_parse_log_level_normalizes_whitespace_and_case() -> None:
    assert parse_log_level(" info ") == logging.INFO


def test_parse_log_level_rejects_empty_value() -> None:
    with pytest.raises(
        LoggingConfigError,
        match="Log level cannot be empty",
    ):
        parse_log_level("   ")


def test_parse_log_level_rejects_unknown_value() -> None:
    with pytest.raises(
        LoggingConfigError,
        match="Unsupported log level",
    ):
        parse_log_level("VERBOSE")


def test_build_formatter_returns_text_formatter() -> None:
    formatter = build_formatter("text")

    assert isinstance(formatter, logging.Formatter)


def test_build_formatter_returns_json_formatter() -> None:
    formatter = build_formatter("json")

    assert isinstance(formatter, JsonLogFormatter)


def test_build_formatter_rejects_unknown_format() -> None:
    with pytest.raises(
        LoggingConfigError,
        match="Unsupported log format",
    ):
        build_formatter("xml")  # type: ignore[arg-type]


def test_json_formatter_outputs_one_line_json() -> None:
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="academic_literature_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Processed %s papers",
        args=(3,),
        exc_info=None,
        func=None,
        sinfo=None,
    )

    output = formatter.format(record)
    payload = json.loads(output)

    assert "\n" not in output
    assert payload["level"] == "INFO"
    assert payload["logger"] == "academic_literature_rag.test"
    assert payload["message"] == "Processed 3 papers"
    assert "timestamp" in payload


def test_configure_logging_sets_root_logger_handler_and_level() -> None:
    configure_logging(
        log_level="WARNING",
        log_format="text",
    )

    root_logger = logging.getLogger()

    assert root_logger.level == logging.WARNING
    assert len(root_logger.handlers) == 1
    assert root_logger.handlers[0].level == logging.WARNING
    assert isinstance(root_logger.handlers[0].formatter, logging.Formatter)


def test_configure_logging_supports_json_format() -> None:
    configure_logging(
        log_level="INFO",
        log_format="json",
    )

    root_logger = logging.getLogger()

    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0].formatter, JsonLogFormatter)