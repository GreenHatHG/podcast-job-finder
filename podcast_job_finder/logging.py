from __future__ import annotations

import logging
from typing import Final

from podcast_job_finder.environment import get_optional_env
from podcast_job_finder.tracing import TraceIdFormatter


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
    normalized_log_level = (
        get_optional_env(LOG_LEVEL_ENV) or DEFAULT_LOG_LEVEL_NAME
    ).upper()

    resolved_log_level = getattr(logging, normalized_log_level, None)
    if not isinstance(resolved_log_level, int):
        return logging.WARNING
    return resolved_log_level
