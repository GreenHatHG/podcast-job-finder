from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.transcription.models import TranscribedSpeechSegment
from podcast_job_finder.transcription.manifest import (
    TranscriptionManifestError,
    parse_transcribed_segment,
)
from podcast_job_finder.audio.segmentation.vad import MAX_FORCED_SPLIT_OVERLAP_MS


JSON_FILE_PATTERN: Final = "*.json"
INPUT_NOT_FOUND_ERROR: Final = "音频转写 JSON 输入不存在：{path}"
EMPTY_DIRECTORY_ERROR: Final = "目录中没有音频转写 JSON：{path}"
INVALID_JSON_OBJECT_ERROR: Final = "音频转写 JSON 必须是对象：{path}"
INVALID_JSON_CONTENT_ERROR: Final = "JSON 中没有有效的音频转写片段：{path}"
READ_JSON_ERROR: Final = "读取音频转写 JSON 失败：{path}，{error_message}"
EMPTY_TRANSCRIPTION_ERROR: Final = "输入中没有可合并的音频转写片段。"
CONFLICTING_SEGMENT_ERROR: Final = (
    "音频转写片段内容冲突：index={index}，start_ms={start_ms}，end_ms={end_ms}"
)
MIXED_INPUT_METADATA_ERROR: Final = "音频转写 JSON 的 {field} 不一致。"
OVERLAPPING_SEGMENTS_ERROR: Final = (
    "音频转写片段时间重叠或顺序错误：previous_index={previous_index}，index={index}"
)
SOURCE_IDENTITY_FIELDS: Final = (
    "source_audio_sha256",
    "eid",
    "source_audio_path",
    "episode_url",
)


class TranscriptionInputError(ValueError):
    """已有音频转写 JSON 无法用于生成文章。"""


@dataclass(slots=True, frozen=True)
class LoadedTranscriptionInput:
    json_paths: tuple[Path, ...]
    title: str
    segments: tuple[TranscribedSpeechSegment, ...]


def load_transcription_inputs(
    input_paths: Sequence[Path],
) -> LoadedTranscriptionInput:
    json_paths = _collect_json_paths(input_paths)
    titles: set[str] = set()
    source_identities: set[tuple[str, str]] = set()
    segments_by_index: dict[int, TranscribedSpeechSegment] = {}
    for path in json_paths:
        payload = _read_json_object(path)
        _collect_input_metadata(
            payload,
            source_identities=source_identities,
        )
        title = payload.get("title")
        if isinstance(title, str) and title.strip():
            titles.add(title.strip())
        for segment in _parse_segments(payload, path):
            _add_segment(segments_by_index, segment)
    _validate_single_metadata_value(source_identities, field="音频来源")
    if not segments_by_index:
        raise TranscriptionInputError(EMPTY_TRANSCRIPTION_ERROR)
    segments = tuple(
        sorted(segments_by_index.values(), key=lambda item: (item.start_ms, item.index))
    )
    _validate_segment_sequence(segments)
    return LoadedTranscriptionInput(
        json_paths=json_paths,
        title=next(iter(titles)) if len(titles) == 1 else "",
        segments=segments,
    )


def _collect_json_paths(input_paths: Sequence[Path]) -> tuple[Path, ...]:
    collected_paths: dict[Path, Path] = {}
    for input_path in input_paths:
        if input_path.is_dir():
            directory_paths = sorted(input_path.glob(JSON_FILE_PATTERN))
            if not directory_paths:
                raise TranscriptionInputError(
                    EMPTY_DIRECTORY_ERROR.format(path=input_path)
                )
            for path in directory_paths:
                collected_paths.setdefault(path.resolve(), path)
            continue
        if not input_path.is_file():
            raise TranscriptionInputError(INPUT_NOT_FOUND_ERROR.format(path=input_path))
        collected_paths.setdefault(input_path.resolve(), input_path)
    return tuple(collected_paths.values())


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        raise TranscriptionInputError(
            READ_JSON_ERROR.format(path=path, error_message=str(error))
        ) from error
    if not isinstance(payload, dict):
        raise TranscriptionInputError(INVALID_JSON_OBJECT_ERROR.format(path=path))
    return payload


def _parse_segments(
    payload: dict[str, object],
    path: Path,
) -> tuple[TranscribedSpeechSegment, ...]:
    try:
        raw_segments = payload.get("segments")
        if isinstance(raw_segments, list):
            return tuple(
                parse_transcribed_segment(item, path=path, index=index)
                for index, item in enumerate(raw_segments)
            )
        if all(field in payload for field in ("index", "start_ms", "end_ms", "text")):
            return (parse_transcribed_segment(payload, path=path, index=0),)
    except TranscriptionManifestError as error:
        raise TranscriptionInputError(str(error)) from error
    raise TranscriptionInputError(INVALID_JSON_CONTENT_ERROR.format(path=path))


def _add_segment(
    segments_by_index: dict[int, TranscribedSpeechSegment],
    segment: TranscribedSpeechSegment,
) -> None:
    existing_segment = segments_by_index.get(segment.index)
    if existing_segment is None:
        segments_by_index[segment.index] = segment
        return
    if existing_segment == segment:
        return
    raise TranscriptionInputError(
        CONFLICTING_SEGMENT_ERROR.format(
            index=segment.index,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
        )
    )


def _collect_input_metadata(
    payload: dict[str, object],
    *,
    source_identities: set[tuple[str, str]],
) -> None:
    for field in SOURCE_IDENTITY_FIELDS:
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            source_identities.add((field, value.strip()))
            break


def _validate_single_metadata_value(
    values: Collection[object],
    *,
    field: str,
) -> None:
    if len(values) <= 1:
        return
    raise TranscriptionInputError(MIXED_INPUT_METADATA_ERROR.format(field=field))


def _validate_segment_sequence(
    segments: Sequence[TranscribedSpeechSegment],
) -> None:
    for previous_segment, segment in pairwise(segments):
        overlap_ms = previous_segment.end_ms - segment.start_ms
        if (
            segment.index > previous_segment.index
            and overlap_ms <= MAX_FORCED_SPLIT_OVERLAP_MS
        ):
            continue
        raise TranscriptionInputError(
            OVERLAPPING_SEGMENTS_ERROR.format(
                previous_index=previous_segment.index,
                index=segment.index,
            )
        )
