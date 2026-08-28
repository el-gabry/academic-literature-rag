from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Literal

LogFormat = Literal[
    "text",
    "json",
]

SUPPORTED_LOG_FORMATS: tuple[str, ...] = (
    "text",
    "json",
)


class LoggingConfigError(ValueError):
    """Raised when logging configuration is invalid."""


class JsonLogFormatter(logging.Formatter):
    """Format log records as one-line JSON objects."""

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:
        """Format one log record as JSON."""

        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=UTC,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            payload["stack"] = record.stack_info

        return json.dumps(
            payload,
            sort_keys=True,
        )


def configure_logging(
    *,
    log_level: str = "INFO",
    log_format: LogFormat = "text",
) -> None:
    """Configure root logging for CLI and application commands."""

    resolved_level = parse_log_level(log_level)
    formatter = build_formatter(log_format)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(resolved_level)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(resolved_level)
    root_logger.addHandler(handler)


def build_formatter(
    log_format: LogFormat,
) -> logging.Formatter:
    """Build a formatter for the requested log format."""

    if log_format == "json":
        return JsonLogFormatter()

    if log_format == "text":
        return logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    raise LoggingConfigError(f"Unsupported log format: {log_format}")


def parse_log_level(
    log_level: str,
) -> int:
    """Parse and validate a logging level name."""

    normalized_level = log_level.strip().upper()

    if not normalized_level:
        raise LoggingConfigError("Log level cannot be empty.")

    level = logging.getLevelName(normalized_level)

    if not isinstance(level, int):
        raise LoggingConfigError(f"Unsupported log level: {log_level}")

    return level