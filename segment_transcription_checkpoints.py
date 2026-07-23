from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from podcast_job_finder.audio.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.transcription import TranscribedSpeechSegment
from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_json,
)
from runtime_signature import build_runtime_signature_hash
from utc_timestamp import build_utc_timestamp


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SegmentTranscriptionCheckpointStore:
    cache_version: int
    runtime_signature: str
    pid: str
    eid: str
    episode_url: str
    title: str | None
    pub_date: str | None

    def load(
        self,
        path: Path,
        *,
        exported_segment: ExportedSpeechSegment,
        previous_text: str,
    ) -> TranscribedSpeechSegment | None:
        if not path.exists():
            return None
        try:
            payload = _read_json_object(path)
            text = _validate_checkpoint_payload(
                payload,
                self._expected_values(exported_segment, previous_text),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.warning(
                "读取音频片段检查点失败，将重新转写：path=%s error=%s",
                path,
                error,
            )
            return None
        return TranscribedSpeechSegment(
            index=exported_segment.index,
            start_ms=exported_segment.segment.start_ms,
            end_ms=exported_segment.segment.end_ms,
            text=text,
        )

    def save(
        self,
        path: Path,
        *,
        exported_segment: ExportedSpeechSegment,
        transcribed_segment: TranscribedSpeechSegment,
        previous_text: str,
    ) -> None:
        payload = {
            "cache_version": self.cache_version,
            "runtime_signature": self.runtime_signature,
            "pid": self.pid,
            "eid": self.eid,
            "title": self.title,
            "pub_date": self.pub_date,
            "episode_url": self.episode_url,
            "created_at": build_utc_timestamp().text,
            "previous_text_signature": _build_previous_text_signature(previous_text),
            "audio_path": str(exported_segment.file_path),
            **transcribed_segment.to_dict(),
        }
        atomic_write_json(
            path,
            payload,
            mode=DEFAULT_FILE_CREATION_MODE,
        )

    def _expected_values(
        self,
        exported_segment: ExportedSpeechSegment,
        previous_text: str,
    ) -> dict[str, object]:
        return {
            "cache_version": self.cache_version,
            "runtime_signature": self.runtime_signature,
            "eid": self.eid,
            "episode_url": self.episode_url,
            "index": exported_segment.index,
            "start_ms": exported_segment.segment.start_ms,
            "end_ms": exported_segment.segment.end_ms,
            "audio_path": str(exported_segment.file_path),
            "previous_text_signature": _build_previous_text_signature(previous_text),
        }


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("音频片段检查点必须是对象。")
    return payload


def _validate_checkpoint_payload(
    payload: dict[str, object],
    expected_values: dict[str, object],
) -> str:
    for field_name, expected_value in expected_values.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(f"音频片段检查点字段 {field_name} 已变化。")
    text = payload.get("text")
    if not isinstance(text, str):
        raise ValueError("音频片段检查点中的 text 必须是字符串。")
    return text


def _build_previous_text_signature(previous_text: str) -> str:
    return build_runtime_signature_hash({"previous_text": previous_text})
