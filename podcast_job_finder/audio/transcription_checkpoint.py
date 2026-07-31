from __future__ import annotations

# pylint: disable=duplicate-code

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from podcast_job_finder.audio.segment_export import (
    ExportedSpeechSegment,
    SegmentAudioFormat,
    build_segment_audio_signature,
)
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.transcription import (
    AudioTranscriberProtocol,
    AudioTranscriptionResult,
    TranscribedSpeechSegment,
    parse_timed_transcription_texts,
    transcribe_speech_segment,
)
from podcast_job_finder.audio.vad import VadConfig
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.runtime_signature import build_runtime_signature_hash
from podcast_job_finder.timestamps import build_utc_timestamp


TRANSCRIPTION_CACHE_VERSION: Final = 5

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SegmentTranscriptionCheckpointStore:
    runtime_signature: str
    metadata: Mapping[str, object]
    expected_metadata: Mapping[str, object]

    def load(
        self,
        path: Path,
        *,
        exported_segment: ExportedSpeechSegment,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscribedSpeechSegment | None:
        if not path.exists():
            return None
        try:
            payload = _read_json_object(path)
            cached_segment = _validate_checkpoint_payload(
                payload,
                self._expected_values(exported_segment, previous_segment),
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
        previous_segment: TranscribedSpeechSegment | None,
    ) -> None:
        payload = {
            **self.metadata,
            "cache_version": TRANSCRIPTION_CACHE_VERSION,
            "runtime_signature": self.runtime_signature,
            "created_at": build_utc_timestamp().text,
            # 当前片段会参考上一片段；上一片段变化时，这个签名会让缓存重新生成。
            "previous_segment_signature": _build_previous_segment_signature(
                previous_segment
            ),
            "audio_path": str(exported_segment.file_path),
            **transcribed_segment.to_dict(),
        }
        atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)

    def _expected_values(
        self,
        exported_segment: ExportedSpeechSegment,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> dict[str, object]:
        return {
            "cache_version": TRANSCRIPTION_CACHE_VERSION,
            "runtime_signature": self.runtime_signature,
            **self.expected_metadata,
            "index": exported_segment.index,
            "start_ms": exported_segment.segment.start_ms,
            "end_ms": exported_segment.segment.end_ms,
            "audio_path": str(exported_segment.file_path),
            "previous_segment_signature": _build_previous_segment_signature(
                previous_segment
            ),
        }


def build_audio_transcription_runtime_signature(
    *,
    transcriber_signature: Mapping[str, object],
    segment_audio_format: SegmentAudioFormat,
    vad_config: VadConfig = VadConfig(),
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
) -> str:
    signature_payload = {
        "cache_version": TRANSCRIPTION_CACHE_VERSION,
        "transcriber": dict(transcriber_signature),
        "vad_config": asdict(vad_config),
        "silence_padding_ms": silence_padding_ms,
        "segment_audio": build_segment_audio_signature(segment_audio_format),
    }
    return build_runtime_signature_hash(signature_payload)


def transcribe_speech_segments_with_checkpoints(
    segments: Sequence[ExportedSpeechSegment],
    *,
    transcriber: AudioTranscriberProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    overwrite: bool = False,
) -> tuple[AudioTranscriptionResult, bool]:
    transcribed_segments: list[TranscribedSpeechSegment] = []
    # 保留上一段的完整结果：下一段既可能用到文字，也可能用到时间信息。
    previous_segment = None
    all_segments_cached = bool(segments)
    for segment in segments:
        transcription_path = segment.file_path.with_suffix(".json")
        transcribed_segment = None
        if not overwrite:
            transcribed_segment = checkpoint_store.load(
                transcription_path,
                exported_segment=segment,
                previous_segment=previous_segment,
            )
        if transcribed_segment is None:
            all_segments_cached = False
            transcribed_segment = transcribe_speech_segment(
                segment,
                transcriber=transcriber,
                previous_segment=previous_segment,
            )
            checkpoint_store.save(
                transcription_path,
                exported_segment=segment,
                transcribed_segment=transcribed_segment,
                previous_segment=previous_segment,
            )
        else:
            logger.info(
                "命中音频片段转写检查点：path=%s index=%d",
                transcription_path,
                segment.index,
            )
        transcribed_segments.append(transcribed_segment)
        previous_segment = transcribed_segment
    return (
        AudioTranscriptionResult(segments=transcribed_segments),
        all_segments_cached,
    )


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
    return TranscribedSpeechSegment(
        index=_require_integer(payload, "index"),
        start_ms=_require_integer(payload, "start_ms"),
        end_ms=_require_integer(payload, "end_ms"),
        text=text,
        character_timestamps=parse_timed_transcription_texts(
            payload.get("character_timestamps"),
            field_name="character_timestamps",
        ),
        sentences=parse_timed_transcription_texts(
            payload.get("sentences"),
            field_name="sentences",
        ),
    )


def _build_previous_segment_signature(
    previous_segment: TranscribedSpeechSegment | None,
) -> str:
    return build_runtime_signature_hash(
        previous_segment.to_dict() if previous_segment is not None else None
    )


def _require_integer(payload: dict[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"音频片段检查点中的 {field_name} 必须是整数。")
    return value
