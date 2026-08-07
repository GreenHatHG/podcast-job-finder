from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence, TypeGuard

from podcast_job_finder.audio.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionResult,
    TranscribedSpeechSegment,
    parse_timed_transcription_texts,
)
from podcast_job_finder.audio.transcription_diagnostics import (
    parse_transcription_diagnostics,
)
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.timestamps import build_utc_timestamp


TRANSCRIPTION_FILE_NAME: Final = "transcription.json"
INVALID_MANIFEST_ERROR: Final = "音频转写清单必须是对象：{path}"
INVALID_SEGMENTS_ERROR: Final = "音频转写清单缺少有效的 segments 数组：{path}"
INVALID_SEGMENT_ERROR: Final = "音频转写清单中的片段无效：{path}，index={index}"
READ_MANIFEST_ERROR: Final = "读取音频转写清单失败：{path}，{error_message}"
SAVE_MANIFEST_ERROR: Final = "保存音频转写清单失败：{path}，{error_message}"
MISSING_EXPORTED_SEGMENT_ERROR: Final = (
    "音频转写结果缺少对应的已导出片段：index={index}"
)


class TranscriptionManifestError(ValueError):
    """保存的音频转写清单无法用于后续处理。"""


@dataclass(slots=True, frozen=True)
class EpisodeTranscriptionManifest:
    title: str
    segments: tuple[TranscribedSpeechSegment, ...]
    metadata: Mapping[str, object]


def save_audio_transcription_manifest(
    path: Path,
    *,
    metadata: Mapping[str, object],
    exported_segments: Sequence[ExportedSpeechSegment],
    result: AudioTranscriptionResult,
) -> dict[str, object]:
    segment_records = _build_segment_records(exported_segments, result)
    payload = {
        **metadata,
        "created_at": build_utc_timestamp().text,
        "segment_count": len(segment_records),
        "text": result.text,
        "segments": segment_records,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)
    except OSError as error:
        raise OSError(
            SAVE_MANIFEST_ERROR.format(path=path, error_message=str(error))
        ) from error
    return payload


def load_episode_transcription_manifest(
    path: Path,
) -> EpisodeTranscriptionManifest:
    payload = _read_manifest_payload(path)
    if not isinstance(payload, dict):
        raise TranscriptionManifestError(INVALID_MANIFEST_ERROR.format(path=path))

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise TranscriptionManifestError(INVALID_SEGMENTS_ERROR.format(path=path))

    raw_title = payload.get("title")
    title = raw_title.strip() if isinstance(raw_title, str) else ""
    return EpisodeTranscriptionManifest(
        title=title,
        segments=tuple(
            parse_transcribed_segment(raw_segment, path=path, index=index)
            for index, raw_segment in enumerate(raw_segments)
        ),
        metadata=payload,
    )


def _build_segment_records(
    exported_segments: Sequence[ExportedSpeechSegment],
    result: AudioTranscriptionResult,
) -> list[dict[str, object]]:
    exported_by_index = {segment.index: segment for segment in exported_segments}
    records: list[dict[str, object]] = []
    for transcribed_segment in result.segments:
        exported_segment = exported_by_index.get(transcribed_segment.index)
        if exported_segment is None:
            raise ValueError(
                MISSING_EXPORTED_SEGMENT_ERROR.format(index=transcribed_segment.index)
            )
        records.append(
            {
                **transcribed_segment.to_dict(),
                "audio_path": str(exported_segment.file_path),
                "transcription_path": str(
                    exported_segment.file_path.with_suffix(".json")
                ),
            }
        )
    return records


def _read_manifest_payload(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        raise TranscriptionManifestError(
            READ_MANIFEST_ERROR.format(path=path, error_message=str(error))
        ) from error


def parse_transcribed_segment(
    payload: object,
    *,
    path: Path,
    index: int,
) -> TranscribedSpeechSegment:
    if not isinstance(payload, dict):
        raise _build_invalid_segment_error(path, index)

    segment_index = payload.get("index")
    start_ms = payload.get("start_ms")
    end_ms = payload.get("end_ms")
    text = payload.get("text")
    if not _is_integer(segment_index):
        raise _build_invalid_segment_error(path, index)
    if not _is_integer(start_ms) or not _is_integer(end_ms):
        raise _build_invalid_segment_error(path, index)
    if start_ms < 0 or end_ms <= start_ms:
        raise _build_invalid_segment_error(path, index)
    if not isinstance(text, str) or not text.strip():
        raise _build_invalid_segment_error(path, index)
    try:
        return TranscribedSpeechSegment(
            index=segment_index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text.strip(),
            character_timestamps=parse_timed_transcription_texts(
                payload.get("character_timestamps"),
                field_name="character_timestamps",
            ),
            sentences=parse_timed_transcription_texts(
                payload.get("sentences"),
                field_name="sentences",
            ),
            diagnostics=parse_transcription_diagnostics(payload.get("diagnostics")),
        )
    except ValueError as error:
        raise _build_invalid_segment_error(path, index) from error


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _build_invalid_segment_error(
    path: Path,
    index: int,
) -> TranscriptionManifestError:
    return TranscriptionManifestError(
        INVALID_SEGMENT_ERROR.format(path=path, index=index)
    )
