"""Audio processing primitives."""

from podcast_job_finder.audio.file_segments import (
    AudioFileDecodeError,
    detect_speech_segments_from_file,
)
from podcast_job_finder.audio.vad import (
    SpeechSegment,
    VadConfig,
    detect_speech_segments,
)

__all__ = [
    "AudioFileDecodeError",
    "SpeechSegment",
    "VadConfig",
    "detect_speech_segments",
    "detect_speech_segments_from_file",
]
