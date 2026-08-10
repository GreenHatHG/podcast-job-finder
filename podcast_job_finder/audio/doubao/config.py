from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from podcast_job_finder.audio.firered_alignment import FireRedAlignmentConfig
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.transcription_diagnostics import (
    LOW_CHARACTER_CONFIDENCE_THRESHOLD,
    MAX_UNCOVERED_SPEECH_MS,
    MIN_SPEECH_COVERAGE_RATIO,
)
from podcast_job_finder.audio.vad import DEFAULT_VAD_THRESHOLD


DOUBAO_MODEL_NAME: Final = "doubao-ime-asr"
DOUBAO_PROTOCOL_VERSION: Final = "apk-1.3.17-twopass"
DOUBAO_MAX_ATTEMPTS: Final = 2
DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS: Final = 30
DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS: Final = 1.0
DOUBAO_MAX_IN_FLIGHT_REQUESTS_METADATA_KEY: Final = "max_in_flight_requests"
DOUBAO_REQUEST_INTERVAL_METADATA_KEY: Final = "request_interval_seconds"


@dataclass(slots=True, frozen=True)
class DoubaoTranscriberConfig:
    alignment_config: FireRedAlignmentConfig
    max_in_flight_requests: int = DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS
    request_interval_seconds: float = DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS
    vad_threshold: float = DEFAULT_VAD_THRESHOLD

    def __post_init__(self) -> None:
        if self.max_in_flight_requests <= 0:
            raise ValueError("DOUBAO_ASR_MAX_IN_FLIGHT_REQUESTS 必须大于 0。")
        if (
            not math.isfinite(self.request_interval_seconds)
            or self.request_interval_seconds <= 0
        ):
            raise ValueError("request_interval_seconds 必须是大于 0 的有限数值。")
        if self.silence_padding_ms < 0:
            raise ValueError("silence_padding_ms 必须大于等于 0。")
        if not 0 < self.vad_threshold < 1:
            raise ValueError("vad_threshold 必须大于 0 且小于 1。")

    def metadata(self) -> dict[str, object]:
        return {
            "transcription_backend": "doubao",
            "model": DOUBAO_MODEL_NAME,
            "protocol_version": DOUBAO_PROTOCOL_VERSION,
            **self.alignment_config.metadata(),
            "max_uncovered_speech_ms": MAX_UNCOVERED_SPEECH_MS,
            "min_speech_coverage_ratio": MIN_SPEECH_COVERAGE_RATIO,
            "low_character_confidence_threshold": (LOW_CHARACTER_CONFIDENCE_THRESHOLD),
            DOUBAO_MAX_IN_FLIGHT_REQUESTS_METADATA_KEY: (self.max_in_flight_requests),
            DOUBAO_REQUEST_INTERVAL_METADATA_KEY: self.request_interval_seconds,
        }
