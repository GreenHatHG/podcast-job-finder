"""Shared JSON-line protocol helpers for FireRed worker entry points."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Final


READY_STATUS: Final = "ready"
RESULT_STATUS: Final = "result"
ERROR_STATUS: Final = "error"
SHUTDOWN_COMMAND: Final = "shutdown"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def parse_request(line: str, *, request_name: str) -> dict[str, Any]:
    request = json.loads(line)
    if not isinstance(request, dict):
        raise ValueError(f"{request_name} 请求必须是 JSON 对象。")
    return request


def is_shutdown(request: dict[str, Any]) -> bool:
    return request.get("command") == SHUTDOWN_COMMAND


def require_audio_path(request: dict[str, Any], *, request_name: str) -> Path:
    value = request.get("audio_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{request_name} 请求缺少 audio_path。")
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"{request_name} 输入音频不存在：{path}")
    return path


def require_nonnegative_int(
    request: dict[str, Any],
    field: str,
    *,
    request_name: str,
) -> int:
    value = request.get(field, 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{request_name} 请求中的 {field} 必须是非负整数。")
    return value


def require_text(request: dict[str, Any], *, request_name: str) -> str:
    value = request.get("text")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{request_name} 请求缺少 text。")
    return value


def write_response(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)
