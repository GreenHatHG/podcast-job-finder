from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from podcast_job_finder.audio.segment_export import (
    SEGMENT_AUDIO_BIT_RATE,
    SEGMENT_AUDIO_CODEC,
    SEGMENT_AUDIO_FORMAT,
    ExportedSpeechSegment,
)
from podcast_job_finder.audio._pcm import PCM_CHANNELS
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.transcription import (
    TRANSCRIPTION_PROMPT_TEMPLATE,
    AudioTranscriptionClientProtocol,
    AudioTranscriptionResult,
    TranscribedSpeechSegment,
    transcribe_speech_segment,
)
from podcast_job_finder.audio.vad import VAD_SAMPLE_RATE, VadConfig
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.llm import LlmRetryConfig, OpenAiCompatibleConfig
from podcast_job_finder.runtime_signature import build_runtime_signature_hash
from podcast_job_finder.timestamps import build_utc_timestamp


TRANSCRIPTION_CACHE_VERSION: Final = 4

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
                "读取音频片段检查点失败，将重新转写：path=%s error=%s", path, error
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
            **self.metadata,
            "cache_version": TRANSCRIPTION_CACHE_VERSION,
            "runtime_signature": self.runtime_signature,
            "created_at": build_utc_timestamp().text,
            "previous_text_signature": _build_previous_text_signature(previous_text),
            "audio_path": str(exported_segment.file_path),
            **transcribed_segment.to_dict(),
        }
        atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)

    def _expected_values(
        self,
        exported_segment: ExportedSpeechSegment,
        previous_text: str,
    ) -> dict[str, object]:
        return {
            "cache_version": TRANSCRIPTION_CACHE_VERSION,
            "runtime_signature": self.runtime_signature,
            **self.expected_metadata,
            "index": exported_segment.index,
            "start_ms": exported_segment.segment.start_ms,
            "end_ms": exported_segment.segment.end_ms,
            "audio_path": str(exported_segment.file_path),
            "previous_text_signature": _build_previous_text_signature(previous_text),
        }


def build_audio_transcription_runtime_signature(
    *,
    llm_config: OpenAiCompatibleConfig,
    vad_config: VadConfig = VadConfig(),
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
) -> str:
    signature_payload = {
        "cache_version": TRANSCRIPTION_CACHE_VERSION,
        "model": llm_config.model,
        "base_url": llm_config.base_url,
        "api_style": llm_config.api_style,
        "prompt_template": TRANSCRIPTION_PROMPT_TEMPLATE,
        "vad_config": asdict(vad_config),
        "silence_padding_ms": silence_padding_ms,
        "segment_audio": {
            "format": SEGMENT_AUDIO_FORMAT,
            "codec": SEGMENT_AUDIO_CODEC,
            "bit_rate": SEGMENT_AUDIO_BIT_RATE,
            "sample_rate": VAD_SAMPLE_RATE,
            "channels": PCM_CHANNELS,
        },
    }
    return build_runtime_signature_hash(signature_payload)


def transcribe_speech_segments_with_checkpoints(
    segments: Sequence[ExportedSpeechSegment],
    *,
    llm_client: AudioTranscriptionClientProtocol,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    retry_config: LlmRetryConfig | None = None,
    overwrite: bool = False,
) -> tuple[AudioTranscriptionResult, bool]:
    transcribed_segments: list[TranscribedSpeechSegment] = []
    previous_text = ""
    all_segments_cached = bool(segments)
    for segment in segments:
        transcription_path = segment.file_path.with_suffix(".json")
        transcribed_segment = None
        if not overwrite:
            transcribed_segment = checkpoint_store.load(
                transcription_path,
                exported_segment=segment,
                previous_text=previous_text,
            )
        if transcribed_segment is None:
            all_segments_cached = False
            transcribed_segment = transcribe_speech_segment(
                segment,
                llm_client=llm_client,
                previous_text=previous_text,
                retry_config=retry_config,
            )
            checkpoint_store.save(
                transcription_path,
                exported_segment=segment,
                transcribed_segment=transcribed_segment,
                previous_text=previous_text,
            )
        else:
            logger.info(
                "命中音频片段转写检查点：path=%s index=%d",
                transcription_path,
                segment.index,
            )
        transcribed_segments.append(transcribed_segment)
        previous_text = transcribed_segment.text
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
