"""Audio processing primitives."""

from podcast_job_finder.audio.file_segments import (
    AudioFileDecodeError,
    detect_speech_segments_from_file,
)
from podcast_job_finder.audio.segment_export import (
    AudioSegmentExportError,
    ExportedSpeechSegment,
    export_speech_segments,
)
from podcast_job_finder.audio.vad import (
    SpeechSegment,
    VadConfig,
    detect_speech_segments,
)

__all__ = [
    "AudioFileDecodeError",
    "AudioSegmentExportError",
    "ExportedSpeechSegment",
    "SpeechSegment",
    "VadConfig",
    "detect_speech_segments",
    "detect_speech_segments_from_file",
    "export_speech_segments",
]
