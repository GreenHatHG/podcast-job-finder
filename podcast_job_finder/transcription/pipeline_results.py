"""音频转写流水线的共享结果类型和状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"


@dataclass(slots=True, frozen=True)
class BatchAudioTranscriptionResult:
    episode_results: list[dict[str, object]]
    success_count: int
    fail_count: int
