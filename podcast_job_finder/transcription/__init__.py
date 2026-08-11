"""转写核心公开接口。"""

from podcast_job_finder.transcription.models import (
    AudioTranscriberProtocol,
    AudioTranscriptionError,
    AudioTranscriptionResult,
    BatchAudioTranscriberProtocol,
    CharacterAlignment,
    TimedTranscriptionText,
    TranscribedSpeechSegment,
    TranscriptionOutput,
    TranscriptionSegmentResult,
)

__all__ = [
    "AudioTranscriptionError",
    "AudioTranscriberProtocol",
    "BatchAudioTranscriberProtocol",
    "AudioTranscriptionResult",
    "TranscriptionOutput",
    "TranscribedSpeechSegment",
    "TranscriptionSegmentResult",
    "TimedTranscriptionText",
    "CharacterAlignment",
]
