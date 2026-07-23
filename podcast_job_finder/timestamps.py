from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final


UTC_TEXT_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
UTC_FILE_LABEL_FORMAT: Final = "%Y%m%d_%H%M%S"


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
