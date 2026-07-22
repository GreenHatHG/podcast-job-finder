"""Audio processing public API."""

from podcast_job_finder.audio.normalized_audio import AudioFileDecodeError
from podcast_job_finder.audio.segment_export import (
    AudioSegmentExportError,
    ExportedSpeechSegment,
)
from podcast_job_finder.audio.speech_pipeline import (
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.vad import (
    SpeechSegment,
    VadConfig,
)

__all__ = [
    "AudioFileDecodeError",
    "AudioSegmentExportError",
    "ExportedSpeechSegment",
    "SpeechSegment",
    "VadConfig",
    "detect_and_export_speech_segments",
]
