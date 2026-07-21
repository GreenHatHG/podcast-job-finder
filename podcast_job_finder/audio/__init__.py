"""Audio processing primitives."""

from podcast_job_finder.audio.vad import (
    SpeechSegment,
    VadConfig,
    detect_speech_segments,
)

__all__ = [
    "SpeechSegment",
    "VadConfig",
    "detect_speech_segments",
]
