from __future__ import annotations

import json
import logging
from typing import Final


DEBUG_REQUEST_TEMPLATE: Final = "xyz 请求 path=%s payload=%s"
DEBUG_RESPONSE_TEMPLATE: Final = "xyz 响应 path=%s status=%s body=%s"
DEBUG_PARSE_FAILURE_TEMPLATE: Final = (
    "xyz 响应无法解析为 JSON path=%s status=%s body=%s"
)
DEBUG_TEXT_TRUNCATION_SUFFIX: Final = "...<truncated>"
MAX_DEBUG_TEXT_LENGTH: Final = 4000
MASKED_VALUE: Final = "***"
SENSITIVE_FIELD_NAMES: Final = frozenset(
    {
        "mobilePhoneNumber",
        "verifyCode",
        "x-jike-access-token",
        "x-jike-refresh-token",
    }
)


def log_request(logger: logging.Logger, *, path: str, payload: object) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(DEBUG_REQUEST_TEMPLATE, path, format_payload(payload))


def log_response(
    logger: logging.Logger,
    *,
    path: str,
    status_code: int,
    payload: object,
) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            DEBUG_RESPONSE_TEMPLATE,
            path,
            status_code,
            format_payload(payload),
        )


def log_parse_failure(
    logger: logging.Logger,
    *,
    path: str,
    status_code: int,
    response_text: str,
) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            DEBUG_PARSE_FAILURE_TEMPLATE,
            path,
            status_code,
            truncate_text(response_text),
        )


def format_payload(payload: object) -> str:
    sanitized_payload = _sanitize_value(payload)
    try:
        serialized_payload = json.dumps(
            sanitized_payload,
            ensure_ascii=False,
            sort_keys=True,
        )
    except TypeError:
        serialized_payload = repr(sanitized_payload)
    return truncate_text(serialized_payload)


def truncate_text(text: str) -> str:
    if len(text) <= MAX_DEBUG_TEXT_LENGTH:
        return text
    truncated_length = MAX_DEBUG_TEXT_LENGTH - len(DEBUG_TEXT_TRUNCATION_SUFFIX)
    return f"{text[:truncated_length]}{DEBUG_TEXT_TRUNCATION_SUFFIX}"


def _sanitize_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: MASKED_VALUE
            if isinstance(key, str) and key in SENSITIVE_FIELD_NAMES
            else _sanitize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return value
