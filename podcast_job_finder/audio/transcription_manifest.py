from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from podcast_job_finder.audio.transcription import TranscribedSpeechSegment


INVALID_MANIFEST_ERROR: Final = "音频转写清单必须是对象：{path}"
INVALID_SEGMENTS_ERROR: Final = "音频转写清单缺少有效的 segments 数组：{path}"
INVALID_SEGMENT_ERROR: Final = "音频转写清单中的片段无效：{path}，index={index}"
READ_MANIFEST_ERROR: Final = "读取音频转写清单失败：{path}，{error_message}"


class TranscriptionManifestError(ValueError):
    """保存的音频转写清单无法用于后续处理。"""


@dataclass(slots=True, frozen=True)
class EpisodeTranscriptionManifest:
    title: str
    segments: tuple[TranscribedSpeechSegment, ...]


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
            _parse_transcribed_segment(raw_segment, path=path, index=index)
            for index, raw_segment in enumerate(raw_segments)
        ),
    )


def _read_manifest_payload(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        raise TranscriptionManifestError(
            READ_MANIFEST_ERROR.format(path=path, error_message=str(error))
        ) from error


def _parse_transcribed_segment(
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
    if (
        not isinstance(segment_index, int)
        or not isinstance(start_ms, int)
        or not isinstance(end_ms, int)
        or not isinstance(text, str)
        or not text.strip()
    ):
        raise _build_invalid_segment_error(path, index)
    return TranscribedSpeechSegment(
        index=segment_index,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text.strip(),
    )


def _build_invalid_segment_error(
    path: Path,
    index: int,
) -> TranscriptionManifestError:
    return TranscriptionManifestError(
        INVALID_SEGMENT_ERROR.format(path=path, index=index)
    )
