from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final, Mapping, Sequence

from podcast_job_finder.audio.segmentation.segment_export import (
    ExportedSpeechSegment,
    SegmentAudioFormat,
    SpeechSegmentExportConfig,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.segmentation.vad import SpeechSegment, VadConfig
from podcast_job_finder.filesystem import DEFAULT_FILE_CREATION_MODE, atomic_write_json
from podcast_job_finder.timestamps import build_utc_timestamp


SPEECH_SEGMENT_CHECKPOINT_FILE_NAME: Final = "speech_segments.json"

logger = logging.getLogger(__name__)


def restore_or_export_speech_segments(  # pylint: disable=too-many-arguments
    *,
    source_path: Path,
    output_dir: Path,
    checkpoint_path: Path,
    vad_config: VadConfig,
    export_config: SpeechSegmentExportConfig,
    resume: bool = False,
) -> list[ExportedSpeechSegment]:
    restored_segments = None
    if resume:
        restored_segments = load_speech_segment_checkpoint(
            checkpoint_path,
            source_path=source_path,
            output_dir=output_dir,
            audio_format=export_config.audio_format,
        )
    if restored_segments is not None:
        logger.info(
            "命中语音切分检查点，跳过 VAD：source_path=%s segment_count=%d",
            source_path,
            len(restored_segments),
        )
        return restored_segments

    restored_segments = detect_and_export_speech_segments(
        source_path,
        output_dir=output_dir,
        config=vad_config,
        export_config=export_config,
    )
    save_speech_segment_checkpoint(
        checkpoint_path,
        source_path=source_path,
        segments=restored_segments,
    )
    return restored_segments


def load_speech_segment_checkpoint(
    path: Path,
    *,
    source_path: Path,
    output_dir: Path,
    audio_format: SegmentAudioFormat,
) -> list[ExportedSpeechSegment] | None:
    if not path.is_file():
        return None
    try:
        payload = _read_json_object(path)
        _validate_checkpoint_metadata(
            payload,
            source_path=source_path,
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
    segments: Sequence[ExportedSpeechSegment],
) -> None:
    payload = {
        "created_at": build_utc_timestamp().text,
        "source_audio_path": str(source_path.resolve()),
        "segment_count": len(segments),
        "segments": [_build_segment_record(segment) for segment in segments],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)


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
) -> None:
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise ValueError(f"源音频文件无效：{source_path}")
    source_audio_path = payload.get("source_audio_path")
    if isinstance(source_audio_path, str):
        if source_audio_path != str(source_path.resolve()):
            raise ValueError("语音切分检查点字段 source_audio_path 已变化。")
        return

    legacy_source = payload.get("source")
    if not isinstance(legacy_source, Mapping):
        raise ValueError("语音切分检查点缺少源音频信息。")
    expected_source = {
        "path": str(source_path.resolve()),
        "size_bytes": source_path.stat().st_size,
        "modified_at_ns": source_path.stat().st_mtime_ns,
    }
    if any(
        legacy_source.get(field_name) != expected_value
        for field_name, expected_value in expected_source.items()
    ):
        raise ValueError("语音切分检查点中的源音频信息已变化。")


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


def _require_integer(payload: Mapping[str, object], field_name: str) -> int:
    value = payload.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"语音切分检查点中的 {field_name} 必须是整数。")
    return value
