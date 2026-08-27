from __future__ import annotations

# pylint: disable=duplicate-code

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.timestamps import build_utc_timestamp
from podcast_job_finder.transcription.models import (
    TranscribedSpeechSegment,
    build_transcribed_speech_segment_from_payload,
)


logger = logging.getLogger("podcast_job_finder.transcription.checkpoint")


@dataclass(slots=True, frozen=True)
class SegmentTranscriptionCheckpointStore:
    metadata: Mapping[str, object]
    expected_metadata: Mapping[str, object]

    def load(
        self,
        path: Path,
        *,
        exported_segment: ExportedSpeechSegment,
    ) -> TranscribedSpeechSegment | None:
        if not path.exists():
            return None
        try:
            payload = _read_json_object(path)
            cached_segment = _validate_checkpoint_payload(
                payload,
                self._expected_values(exported_segment),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.warning(
                "读取音频片段检查点失败，将重新转写：path=%s error=%s", path, error
            )
            return None
        return cached_segment

    def save(
        self,
        path: Path,
        *,
        exported_segment: ExportedSpeechSegment,
        transcribed_segment: TranscribedSpeechSegment,
    ) -> None:
        payload = {
            **self.metadata,
            "created_at": build_utc_timestamp().text,
            "audio_path": str(exported_segment.file_path),
            **transcribed_segment.to_dict(),
        }
        atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)

    def _expected_values(
        self,
        exported_segment: ExportedSpeechSegment,
    ) -> dict[str, object]:
        expected_values = {
            **self.expected_metadata,
            "index": exported_segment.index,
            "start_ms": exported_segment.segment.start_ms,
            "end_ms": exported_segment.segment.end_ms,
        }
        return expected_values


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("音频片段检查点必须是对象。")
    return payload


def _validate_checkpoint_payload(
    payload: dict[str, object],
    expected_values: dict[str, object],
) -> TranscribedSpeechSegment:
    for field_name, expected_value in expected_values.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(f"音频片段检查点字段 {field_name} 已变化。")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("音频片段检查点中的 text 必须是字符串。")
    # 完整恢复文字和时间信息，缓存片段才能继续作为下一段的参考。
    return build_transcribed_speech_segment_from_payload(
        payload=payload,
        index=_require_integer(payload, "index"),
        start_ms=_require_integer(payload, "start_ms"),
        end_ms=_require_integer(payload, "end_ms"),
        text=text,
    )


def _require_integer(payload: dict[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"音频片段检查点中的 {field_name} 必须是整数。")
    return value
