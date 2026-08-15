from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final


UTC_TEXT_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
UTC_FILE_LABEL_FORMAT: Final = "%Y%m%d_%H%M%S"
MILLISECONDS_PER_SECOND: Final = 1_000
SECONDS_PER_MINUTE: Final = 60
MINUTES_PER_HOUR: Final = 60


@dataclass(slots=True, frozen=True)
class UtcTimestamp:
    text: str
    file_label: str


def build_utc_timestamp() -> UtcTimestamp:
    now = datetime.now(tz=timezone.utc)
    return UtcTimestamp(
        text=now.strftime(UTC_TEXT_FORMAT),
        file_label=now.strftime(UTC_FILE_LABEL_FORMAT),
    )


def format_duration_ms(duration_ms: int) -> str:
    """把毫秒时长转换为 HH:MM:SS.mmm。"""
    total_seconds, milliseconds = divmod(duration_ms, MILLISECONDS_PER_SECOND)
    total_minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    hours, minutes = divmod(total_minutes, MINUTES_PER_HOUR)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
