from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.pcm_decode import (
    AudioFileDecodeError,
    decode_audio_file,
)
from podcast_job_finder.audio.vad import VAD_SAMPLE_RATE, SpeechSegment


SEGMENT_FILE_NAME_TEMPLATE: Final = "segment_{index:04d}_{start_time}_{end_time}.wav"
PARTIAL_FILE_SUFFIX: Final = ".part"
DEFAULT_SILENCE_PADDING_MS: Final = 500
MILLISECONDS_PER_SECOND: Final = 1_000
SECONDS_PER_MINUTE: Final = 60
MINUTES_PER_HOUR: Final = 60
# s16le PCM 的每个采样点占 16 位，即 2 字节；写 WAV 头和生成静音时都使用该值。
PCM_SAMPLE_WIDTH_BYTES: Final = 2
INVALID_SILENCE_PADDING_ERROR: Final = "silence_padding_ms 必须大于等于 0。"
INVALID_SEGMENT_ERROR: Final = "音频片段时间无效：start_ms={start_ms}，end_ms={end_ms}"
OUTPUT_DIR_ERROR: Final = "无法创建音频片段目录：{path}，{error_message}"
EXISTING_SEGMENT_ERROR: Final = "音频片段文件已经存在：{path}"
DECODE_FOR_EXPORT_ERROR: Final = "导出音频片段前解码失败：{error_message}"
EXPORT_SEGMENT_ERROR: Final = "无法导出音频片段：{path}，{error_message}"
EMPTY_SEGMENT_ERROR: Final = "导出的音频片段没有声音数据：{path}"
PUBLISH_SEGMENT_ERROR: Final = "无法保存音频片段：{path}，{error_message}"


class AudioSegmentExportError(RuntimeError):
    """音频片段无法导出时抛出的错误。"""


@dataclass(slots=True, frozen=True)
class ExportedSpeechSegment:
    index: int
    segment: SpeechSegment
    file_path: Path

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            **self.segment.to_dict(),
            "file_path": str(self.file_path),
        }


def export_speech_segments(
    audio_path: Path,
    segments: Sequence[SpeechSegment],
    *,
    output_dir: Path,
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
    overwrite: bool = False,
) -> list[ExportedSpeechSegment]:
    """按时间段导出 WAV 文件，并在每个文件前后添加静音。

    静音为后续语音识别留出开始和结束缓冲。导出的文件统一为 16 kHz
    单声道 WAV，文件名同时包含片段编号、开始时间和结束时间。
    """
    if silence_padding_ms < 0:
        raise ValueError(INVALID_SILENCE_PADDING_ERROR)
    _prepare_output_dir(output_dir)

    exported_segments = _prepare_exported_segments(
        segments,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if not exported_segments:
        return []

    samples = _decode_audio_for_export(audio_path)
    silence = _build_silence(silence_padding_ms)
    for exported_segment in exported_segments:
        _export_segment_file(
            samples,
            exported_segment.segment,
            target_path=exported_segment.file_path,
            silence=silence,
        )
    return exported_segments


def _prepare_exported_segments(
    segments: Sequence[SpeechSegment],
    *,
    output_dir: Path,
    overwrite: bool,
) -> list[ExportedSpeechSegment]:
    exported_segments = []
    for index, segment in enumerate(segments, start=1):
        _validate_segment(segment)
        target_path = output_dir / SEGMENT_FILE_NAME_TEMPLATE.format(
            index=index,
            start_time=_format_file_timestamp(segment.start_ms),
            end_time=_format_file_timestamp(segment.end_ms),
        )
        if target_path.exists() and not overwrite:
            raise AudioSegmentExportError(
                EXISTING_SEGMENT_ERROR.format(path=target_path)
            )
        exported_segments.append(
            ExportedSpeechSegment(
                index=index,
                segment=segment,
                file_path=target_path,
            )
        )
    return exported_segments


def _prepare_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise AudioSegmentExportError(
            OUTPUT_DIR_ERROR.format(path=output_dir, error_message=str(error))
        ) from error


def _validate_segment(segment: SpeechSegment) -> None:
    if segment.start_ms < 0 or segment.end_ms <= segment.start_ms:
        raise ValueError(
            INVALID_SEGMENT_ERROR.format(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
            )
        )


def _export_segment_file(
    samples: NDArray[np.int16],
    segment: SpeechSegment,
    *,
    target_path: Path,
    silence: bytes,
) -> None:
    partial_path = target_path.with_suffix(f"{target_path.suffix}{PARTIAL_FILE_SUFFIX}")
    partial_path.unlink(missing_ok=True)
    try:
        segment_samples = _slice_segment(samples, segment, target_path=target_path)
        _write_wav_file(partial_path, segment_samples, silence=silence)
        _publish_segment_file(partial_path, target_path)
    finally:
        partial_path.unlink(missing_ok=True)


def _decode_audio_for_export(audio_path: Path) -> NDArray[np.int16]:
    try:
        return decode_audio_file(
            audio_path,
            sample_rate=VAD_SAMPLE_RATE,
        )
    except AudioFileDecodeError as error:
        raise AudioSegmentExportError(
            DECODE_FOR_EXPORT_ERROR.format(error_message=str(error))
        ) from error


def _build_silence(duration_ms: int) -> bytes:
    sample_count = _milliseconds_to_sample_index(duration_ms)
    return bytes(sample_count * PCM_SAMPLE_WIDTH_BYTES)


def _slice_segment(
    samples: NDArray[np.int16],
    segment: SpeechSegment,
    *,
    target_path: Path,
) -> NDArray[np.int16]:
    start_sample = _milliseconds_to_sample_index(segment.start_ms)
    end_sample = _milliseconds_to_sample_index(segment.end_ms)
    segment_samples = samples[start_sample:end_sample]
    if segment_samples.size == 0:
        raise AudioSegmentExportError(EMPTY_SEGMENT_ERROR.format(path=target_path))
    return segment_samples


def _write_wav_file(
    output_path: Path,
    samples: NDArray[np.int16],
    *,
    silence: bytes,
) -> None:
    try:
        with wave.Wave_write(str(output_path)) as output_file:
            output_file.setnchannels(1)
            output_file.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
            output_file.setframerate(VAD_SAMPLE_RATE)
            output_file.writeframesraw(silence)
            output_file.writeframesraw(samples.tobytes())
            output_file.writeframes(silence)
    except (OSError, wave.Error) as error:
        raise AudioSegmentExportError(
            EXPORT_SEGMENT_ERROR.format(
                path=output_path,
                error_message=str(error),
            )
        ) from error


def _publish_segment_file(partial_path: Path, target_path: Path) -> None:
    try:
        partial_path.replace(target_path)
    except OSError as error:
        raise AudioSegmentExportError(
            PUBLISH_SEGMENT_ERROR.format(
                path=target_path,
                error_message=str(error),
            )
        ) from error


def _milliseconds_to_sample_index(duration_ms: int) -> int:
    return round(duration_ms * VAD_SAMPLE_RATE / MILLISECONDS_PER_SECOND)


def _format_file_timestamp(timestamp_ms: int) -> str:
    """把毫秒转换为适合放进文件名的“时-分-秒.毫秒”格式。"""
    total_seconds, milliseconds = divmod(timestamp_ms, MILLISECONDS_PER_SECOND)
    total_minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    hours, minutes = divmod(total_minutes, MINUTES_PER_HOUR)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}.{milliseconds:03d}"
