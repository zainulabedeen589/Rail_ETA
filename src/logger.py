"""Structured logging configuration for Rail ETA.

Provides a ``setup_logging`` helper that configures the root logger with a
JSON-friendly formatter when running under a CI/CD pipeline or a human
readable formatter for local development.
"""

import logging
import logging.config
import sys
from typing import Optional

from config import settings


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        import json

        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: Optional[str] = None) -> None:
    """Initialise application-wide logging.

    Args:
        level: Override the log level (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = (level or "INFO").upper()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": _JsonFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "standard",
                "level": log_level,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": log_level,
        },
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    return logging.getLogger(name)
