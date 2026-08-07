from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.audio.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionResult,
    TranscribedSpeechSegment,
)
from podcast_job_finder.audio.transcription_diagnostics import (
    LOW_CHARACTER_CONFIDENCE_THRESHOLD,
    TranscriptionAttemptDiagnostics,
)
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.timestamps import build_utc_timestamp


TRANSCRIPTION_QUALITY_REPORT_FILE_NAME: Final = "transcription_quality_report.json"
MISSING_EXPORTED_SEGMENT_ERROR: Final = "质量报告缺少音频片段：index={index}"


def save_transcription_quality_report(
    path: Path,
    result: AudioTranscriptionResult,
    *,
    exported_segments: Sequence[ExportedSpeechSegment],
) -> dict[str, object]:
    exported_by_index = {segment.index: segment for segment in exported_segments}
    segment_records = [
        record
        for segment in result.segments
        if (
            record := _build_segment_record(
                segment,
                exported_by_index=exported_by_index,
            )
        )
        is not None
    ]
    evaluated_character_count = sum(
        1
        for segment in result.segments
        for item in segment.character_timestamps
        if item.confidence is not None
    )
    suspicious_character_count = sum(
        1
        for segment in result.segments
        for item in segment.character_timestamps
        if item.confidence is not None
        and item.confidence < LOW_CHARACTER_CONFIDENCE_THRESHOLD
    )
    inaccurate_segment_count = sum(
        1 for record in segment_records if record["suspicious_characters"]
    )
    truncation_segment_count = sum(
        1 for record in segment_records if record["has_truncation_warning"]
    )
    payload = {
        "created_at": build_utc_timestamp().text,
        "low_character_confidence_threshold": (LOW_CHARACTER_CONFIDENCE_THRESHOLD),
        "segment_count": len(result.segments),
        "evaluated_character_count": evaluated_character_count,
        "suspicious_character_count": suspicious_character_count,
        "inaccurate_segment_count": inaccurate_segment_count,
        "truncation_segment_count": truncation_segment_count,
        "affected_segment_count": len(segment_records),
        "clean_segment_count": len(result.segments) - len(segment_records),
        "segments": segment_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)
    return payload


def _build_segment_record(
    segment: TranscribedSpeechSegment,
    *,
    exported_by_index: dict[int, ExportedSpeechSegment],
) -> dict[str, object] | None:
    exported_segment = exported_by_index.get(segment.index)
    if exported_segment is None:
        raise ValueError(MISSING_EXPORTED_SEGMENT_ERROR.format(index=segment.index))
    suspicious_characters = [
        item.to_dict()
        for item in segment.character_timestamps
        if item.confidence is not None
        and item.confidence < LOW_CHARACTER_CONFIDENCE_THRESHOLD
    ]
    diagnostics = segment.diagnostics
    has_diagnostic_issue = diagnostics is not None and any(
        attempt.assessment.is_anomalous for attempt in diagnostics.attempts
    )
    if not suspicious_characters and not has_diagnostic_issue:
        return None
    payload: dict[str, object] = {
        "index": segment.index,
        "start_ms": segment.start_ms,
        "end_ms": segment.end_ms,
        "text": segment.text,
        "audio_path": str(exported_segment.file_path),
        "transcription_path": str(exported_segment.file_path.with_suffix(".json")),
        "has_truncation_warning": has_diagnostic_issue,
        "suspicious_characters": suspicious_characters,
    }
    if diagnostics is not None:
        payload["selected_attempt"] = diagnostics.selected_attempt
        payload["attempts"] = [
            _build_attempt_record(attempt) for attempt in diagnostics.attempts
        ]
    return payload


def _build_attempt_record(
    attempt: TranscriptionAttemptDiagnostics,
) -> dict[str, object]:
    payload = {
        "attempt": attempt.attempt,
        "assessment": attempt.assessment.to_dict(),
        "raw_response_count": len(attempt.raw_responses),
    }
    if attempt.error:
        payload["error"] = attempt.error
    return payload
