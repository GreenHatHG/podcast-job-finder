from __future__ import annotations

import logging
import os
from typing import Final

from tracing import TraceIdFormatter


LOG_LEVEL_ENV: Final = "LOG_LEVEL"
DEFAULT_LOG_LEVEL_NAME: Final = "INFO"
LOG_FORMAT: Final = (
    "%(asctime)s %(levelname)s %(name)s [trace_id=%(trace_id)s]: %(message)s"
)


def configure_logging() -> None:
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(_resolve_log_level())

    handler = logging.StreamHandler()
    handler.setFormatter(TraceIdFormatter(LOG_FORMAT))
    root_logger.addHandler(handler)


def _resolve_log_level() -> int:
    raw_log_level = os.getenv(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL_NAME)
    normalized_log_level = raw_log_level.strip().upper()
    if not normalized_log_level:
        normalized_log_level = DEFAULT_LOG_LEVEL_NAME

    resolved_log_level = getattr(logging, normalized_log_level, None)
    if not isinstance(resolved_log_level, int):
        return logging.WARNING
    return resolved_log_level
