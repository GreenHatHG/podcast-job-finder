from __future__ import annotations

import logging

from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.transcription.models import (
    AudioTranscriberProtocol,
    PREVIOUS_SEGMENT_ORDER_ERROR,
    TranscribedSpeechSegment,
    TranscriptionOutput,
)


logger = logging.getLogger(__name__)


def transcribe_speech_segment(
    segment: ExportedSpeechSegment,
    *,
    transcriber: AudioTranscriberProtocol,
    previous_segment: TranscribedSpeechSegment | None = None,
) -> TranscribedSpeechSegment:
    validate_previous_segment_order(segment, previous_segment)
    logger.info(
        "识别音频片段：index=%d start_ms=%d end_ms=%d",
        segment.index,
        segment.segment.start_ms,
        segment.segment.end_ms,
    )
    output = transcriber.transcribe(segment, previous_segment=previous_segment)
    return build_transcribed_speech_segment(segment, output)


def validate_previous_segment_order(
    segment: ExportedSpeechSegment,
    previous_segment: TranscribedSpeechSegment | None,
) -> None:
    if previous_segment is None:
        return
    expected_index = segment.index - 1
    if previous_segment.index != expected_index:
        raise ValueError(
            PREVIOUS_SEGMENT_ORDER_ERROR.format(
                expected_index=expected_index,
                actual_index=previous_segment.index,
                current_index=segment.index,
            )
        )


def build_transcribed_speech_segment(
    segment: ExportedSpeechSegment,
    output: TranscriptionOutput,
) -> TranscribedSpeechSegment:
    return TranscribedSpeechSegment(
        index=segment.index,
        start_ms=segment.segment.start_ms,
        end_ms=segment.segment.end_ms,
        text=output.text,
        character_timestamps=output.character_timestamps,
        sentences=output.sentences,
        diagnostics=output.diagnostics,
    )
