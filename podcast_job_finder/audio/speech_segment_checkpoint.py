from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Final, Mapping, Sequence

from podcast_job_finder.audio.segment_export import (
    ExportedSpeechSegment,
    SegmentAudioFormat,
    SpeechSegmentExportConfig,
    build_segment_audio_signature,
)
from podcast_job_finder.audio.speech_pipeline import detect_and_export_speech_segments
from podcast_job_finder.audio.transcription_checkpoint import (
    SegmentTranscriptionCheckpointStore,
)
from podcast_job_finder.audio.vad import VAD_SAMPLE_RATE, SpeechSegment, VadConfig
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.runtime_signature import build_runtime_signature_hash
from podcast_job_finder.timestamps import build_utc_timestamp


SPEECH_SEGMENT_CHECKPOINT_FILE_NAME: Final = "speech_segments.json"
SPEECH_SEGMENT_CHECKPOINT_VERSION: Final = 1
SEGMENT_FILE_PATTERN: Final = re.compile(
    r"^segment_(?P<index>\d+)_(?P<start>\d{2,}-\d{2}-\d{2}\.\d{3})_"
    r"(?P<end>\d{2,}-\d{2}-\d{2}\.\d{3})\.(?P<format>mp3|wav)$"
)
MILLISECONDS_PER_SECOND: Final = 1_000
SECONDS_PER_MINUTE: Final = 60
MINUTES_PER_HOUR: Final = 60

logger = logging.getLogger(__name__)


def build_speech_segment_runtime_signature(
    *,
    vad_config: VadConfig,
    silence_padding_ms: int,
    audio_format: SegmentAudioFormat,
) -> str:
    return build_runtime_signature_hash(
        {
            "cache_version": SPEECH_SEGMENT_CHECKPOINT_VERSION,
            "vad_config": asdict(vad_config),
            "silence_padding_ms": silence_padding_ms,
            "segment_audio": build_segment_audio_signature(audio_format),
        }
    )


def restore_or_export_speech_segments(  # pylint: disable=too-many-arguments
    *,
    source_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    vad_config: VadConfig,
    export_config: SpeechSegmentExportConfig,
    transcription_checkpoint_store: SegmentTranscriptionCheckpointStore,
) -> list[ExportedSpeechSegment]:
    runtime_signature = build_speech_segment_runtime_signature(
        vad_config=vad_config,
        silence_padding_ms=export_config.silence_padding_ms,
        audio_format=export_config.audio_format,
    )
    checkpoint_exists = checkpoint_path.exists()
    restored_segments = load_speech_segment_checkpoint(
        checkpoint_path,
        source_path=source_path,
        output_dir=output_dir,
        runtime_signature=runtime_signature,
        audio_format=export_config.audio_format,
    )
    if restored_segments is not None:
        logger.info(
            "命中语音切分检查点，跳过 VAD：source_path=%s segment_count=%d",
            source_path,
            len(restored_segments),
        )
        return restored_segments

    restored_segments = (
        None
        if checkpoint_exists
        else _restore_legacy_speech_segments(
            output_dir,
            audio_format=export_config.audio_format,
            checkpoint_store=transcription_checkpoint_store,
        )
    )
    if restored_segments is not None:
        logger.info(
            "导入已有语音片段，跳过 VAD：source_path=%s segment_count=%d",
            source_path,
            len(restored_segments),
        )
    else:
        restored_segments = detect_and_export_speech_segments(
            source_path,
            output_dir=output_dir,
            config=vad_config,
            export_config=export_config,
        )
    save_speech_segment_checkpoint(
        checkpoint_path,
        source_path=source_path,
        runtime_signature=runtime_signature,
        segments=restored_segments,
    )
    return restored_segments


def load_speech_segment_checkpoint(
    path: Path,
    *,
    source_path: Path,
    output_dir: Path,
    runtime_signature: str,
    audio_format: SegmentAudioFormat,
) -> list[ExportedSpeechSegment] | None:
    if not path.is_file():
        return None
    try:
        payload = _read_json_object(path)
        _validate_checkpoint_metadata(
            payload,
            source_path=source_path,
            runtime_signature=runtime_signature,
        )
        segments = _parse_segment_records(
            payload.get("segments"),
            output_dir=output_dir,
            audio_format=audio_format,
        )
        if payload.get("segment_count") != len(segments):
            raise ValueError("语音切分检查点中的 segment_count 已变化。")
        return segments
    except (OSError, ValueError, json.JSONDecodeError) as error:
        logger.warning(
            "读取语音切分检查点失败，将重新检测：path=%s error=%s", path, error
        )
        return None


def save_speech_segment_checkpoint(
    path: Path,
    *,
    source_path: Path,
    runtime_signature: str,
    segments: Sequence[ExportedSpeechSegment],
) -> None:
    payload = {
        "cache_version": SPEECH_SEGMENT_CHECKPOINT_VERSION,
        "runtime_signature": runtime_signature,
        "created_at": build_utc_timestamp().text,
        "source": _build_source_fingerprint(source_path),
        "segment_count": len(segments),
        "segments": [_build_segment_record(segment) for segment in segments],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)


def discover_legacy_speech_segments(
    output_dir: Path,
    *,
    audio_format: SegmentAudioFormat,
) -> list[ExportedSpeechSegment]:
    records = []
    for file_path in output_dir.glob(f"segment_*.{audio_format}"):
        match = SEGMENT_FILE_PATTERN.fullmatch(file_path.name)
        if match is None or match.group("format") != audio_format:
            continue
        records.append(_build_legacy_segment(file_path, match.groupdict()))
    records.sort(key=lambda segment: segment.index)
    if records:
        _validate_contiguous_segments(records)
    return records


def _restore_legacy_speech_segments(
    output_dir: Path,
    *,
    audio_format: SegmentAudioFormat,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
) -> list[ExportedSpeechSegment] | None:
    try:
        segments = discover_legacy_speech_segments(
            output_dir,
            audio_format=audio_format,
        )
    except (OSError, ValueError) as error:
        logger.warning(
            "检查已有语音片段失败，将重新检测：output_dir=%s error=%s",
            output_dir,
            error,
        )
        return None
    if not segments:
        return None
    first_segment = segments[0]
    first_checkpoint = checkpoint_store.load(
        first_segment.file_path.with_suffix(".json"),
        exported_segment=first_segment,
        previous_segment=None,
    )
    return segments if first_checkpoint is not None else None


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError("语音切分检查点必须是对象。")
    return payload


def _validate_checkpoint_metadata(
    payload: Mapping[str, object],
    *,
    source_path: Path,
    runtime_signature: str,
) -> None:
    expected_values = {
        "cache_version": SPEECH_SEGMENT_CHECKPOINT_VERSION,
        "runtime_signature": runtime_signature,
        "source": _build_source_fingerprint(source_path),
    }
    for field_name, expected_value in expected_values.items():
        if payload.get(field_name) != expected_value:
            raise ValueError(f"语音切分检查点字段 {field_name} 已变化。")


def _build_source_fingerprint(path: Path) -> dict[str, int | str]:
    resolved_path = path.resolve()
    stat_result = resolved_path.stat()
    if not resolved_path.is_file() or stat_result.st_size <= 0:
        raise ValueError(f"源音频文件无效：{resolved_path}")
    return {
        "path": str(resolved_path),
        "size_bytes": stat_result.st_size,
        "modified_at_ns": stat_result.st_mtime_ns,
    }


def _build_segment_record(segment: ExportedSpeechSegment) -> dict[str, int | str]:
    return {
        "index": segment.index,
        "start_sample": segment.segment.start_sample,
        "end_sample": segment.segment.end_sample,
        "file_path": str(segment.file_path.resolve()),
    }


def _parse_segment_records(
    value: object,
    *,
    output_dir: Path,
    audio_format: SegmentAudioFormat,
) -> list[ExportedSpeechSegment]:
    if not isinstance(value, list):
        raise ValueError("语音切分检查点缺少 segments 数组。")
    segments = [
        _parse_segment_record(
            record,
            output_dir=output_dir,
            audio_format=audio_format,
        )
        for record in value
    ]
    _validate_contiguous_segments(segments)
    return segments


def _parse_segment_record(
    value: object,
    *,
    output_dir: Path,
    audio_format: SegmentAudioFormat,
) -> ExportedSpeechSegment:
    if not isinstance(value, dict):
        raise ValueError("语音切分检查点中的片段必须是对象。")
    index = _require_integer(value, "index")
    start_sample = _require_integer(value, "start_sample")
    end_sample = _require_integer(value, "end_sample")
    file_path_value = value.get("file_path")
    if not isinstance(file_path_value, str) or not file_path_value:
        raise ValueError("语音切分检查点中的 file_path 必须是字符串。")
    file_path = Path(file_path_value).resolve()
    _validate_segment_file(
        file_path,
        output_dir=output_dir,
        audio_format=audio_format,
    )
    return _build_exported_segment(
        index=index,
        start_sample=start_sample,
        end_sample=end_sample,
        file_path=file_path,
    )


def _build_legacy_segment(
    file_path: Path,
    fields: Mapping[str, str],
) -> ExportedSpeechSegment:
    _validate_segment_file(
        file_path,
        output_dir=file_path.parent,
        audio_format=fields["format"],
    )
    start_ms = _parse_file_timestamp(fields["start"])
    end_ms = _parse_file_timestamp(fields["end"])
    return _build_exported_segment(
        index=int(fields["index"]),
        start_sample=_milliseconds_to_samples(start_ms),
        end_sample=_milliseconds_to_samples(end_ms),
        file_path=file_path.resolve(),
    )


def _build_exported_segment(
    *,
    index: int,
    start_sample: int,
    end_sample: int,
    file_path: Path,
) -> ExportedSpeechSegment:
    if index <= 0 or start_sample < 0 or end_sample <= start_sample:
        raise ValueError(
            "语音片段位置无效："
            f"index={index} start_sample={start_sample} end_sample={end_sample}"
        )
    return ExportedSpeechSegment(
        index=index,
        segment=SpeechSegment(start_sample=start_sample, end_sample=end_sample),
        file_path=file_path,
    )


def _validate_segment_file(
    path: Path,
    *,
    output_dir: Path,
    audio_format: str,
) -> None:
    resolved_path = path.resolve()
    if resolved_path.parent != output_dir.resolve():
        raise ValueError(f"语音片段不在预期目录中：{resolved_path}")
    if resolved_path.suffix != f".{audio_format}" or not resolved_path.is_file():
        raise ValueError(f"语音片段文件无效：{resolved_path}")
    if resolved_path.stat().st_size <= 0:
        raise ValueError(f"语音片段文件为空：{resolved_path}")


def _validate_contiguous_segments(segments: Sequence[ExportedSpeechSegment]) -> None:
    for expected_index, segment in enumerate(segments, start=1):
        if segment.index != expected_index:
            raise ValueError(
                "语音片段编号不连续："
                f"expected_index={expected_index} actual_index={segment.index}"
            )


def _parse_file_timestamp(value: str) -> int:
    hours_text, minutes_text, seconds_text = value.split("-")
    seconds, milliseconds = seconds_text.split(".")
    return (
        int(hours_text) * MINUTES_PER_HOUR * SECONDS_PER_MINUTE
        + int(minutes_text) * SECONDS_PER_MINUTE
        + int(seconds)
    ) * MILLISECONDS_PER_SECOND + int(milliseconds)


def _milliseconds_to_samples(value: int) -> int:
    return round(value * VAD_SAMPLE_RATE / MILLISECONDS_PER_SECOND)


def _require_integer(payload: Mapping[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"语音切分检查点中的 {field_name} 必须是整数。")
    return value
