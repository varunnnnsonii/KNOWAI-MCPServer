"""Structured logging for the MCP server.

Provides a configured logger with JSON-style structured output
for production observability. Sensitive data (full query text,
secrets) is never logged.
"""

from __future__ import annotations

import logging
import sys
import json
from datetime import datetime, timezone


class StructuredFormatter(logging.Formatter):
    """JSON-lines log formatter for structured observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge extra fields added via logger.info("msg", extra={...})
        for key in ("event", "duration_ms", "count", "error", "file", "query_length",
                     "results_count", "top_score", "filters", "files_processed",
                     "entries_indexed", "index_size_bytes", "model_name",
                     "retry_count", "error_type", "threshold", "module_name"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a structured logger for the given module name.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
