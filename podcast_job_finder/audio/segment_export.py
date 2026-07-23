from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio._pcm import PCM_SAMPLE_WIDTH_BYTES
from podcast_job_finder.audio.normalized_audio import NormalizedAudio
from podcast_job_finder.audio.vad import VAD_SAMPLE_RATE, SpeechSegment
from podcast_job_finder.filesystem import (
    AtomicWriteConflictError,
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_file,
)


SEGMENT_FILE_NAME_TEMPLATE: Final = "segment_{index:04d}_{start_time}_{end_time}.wav"
MILLISECONDS_PER_SECOND: Final = 1_000
SECONDS_PER_MINUTE: Final = 60
MINUTES_PER_HOUR: Final = 60
INVALID_SEGMENT_ERROR: Final = (
    "音频片段位置无效：start_sample={start_sample}，end_sample={end_sample}"
)
OUTPUT_DIR_ERROR: Final = "无法创建音频片段目录：{path}，{error_message}"
EXISTING_SEGMENT_ERROR: Final = "音频片段文件已经存在：{path}"
EXPORT_SEGMENT_ERROR: Final = "无法导出音频片段：{path}，{error_message}"
EMPTY_SEGMENT_ERROR: Final = "导出的音频片段没有声音数据：{path}"

logger = logging.getLogger(__name__)


class AudioSegmentExportError(RuntimeError):
    """音频片段无法导出时抛出的错误。"""


@dataclass(slots=True, frozen=True)
class ExportedSpeechSegment:
    """记录一个语音片段的导出顺序、时间范围和输出文件路径。"""

    # 从 1 开始的导出顺序，同时用于生成文件名中的片段编号。
    index: int

    # VAD 检测得到的精确采样范围。
    segment: SpeechSegment

    # 片段完成导出后对应的 WAV 文件路径。
    file_path: Path

    def to_dict(self) -> dict[str, int | str]:
        return {
            "index": self.index,
            **self.segment.to_dict(),
            "file_path": str(self.file_path),
        }


def _export_speech_segments(
    audio: NormalizedAudio,
    segments: Sequence[SpeechSegment],
    output_dir: Path,
    silence_padding_ms: int,
    overwrite: bool,
) -> list[ExportedSpeechSegment]:
    """从规范化 WAV 按采样位置导出说话片段。"""
    _prepare_output_dir(output_dir)
    exported_segments = _prepare_exported_segments(
        segments,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    if not exported_segments:
        return []

    silence = _build_silence(silence_padding_ms)
    for exported_segment in exported_segments:
        segment = exported_segment.segment
        samples = audio.read_samples(segment.start_sample, segment.end_sample)
        if samples.size == 0:
            raise AudioSegmentExportError(
                EMPTY_SEGMENT_ERROR.format(path=exported_segment.file_path)
            )
        _export_segment_file(
            samples,
            target_path=exported_segment.file_path,
            silence=silence,
            overwrite=overwrite,
        )
        logger.debug(
            "音频片段已导出：index=%d start_ms=%d end_ms=%d file_path=%s",
            exported_segment.index,
            segment.start_ms,
            segment.end_ms,
            exported_segment.file_path,
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
    if segment.start_sample < 0 or segment.end_sample <= segment.start_sample:
        raise ValueError(
            INVALID_SEGMENT_ERROR.format(
                start_sample=segment.start_sample,
                end_sample=segment.end_sample,
            )
        )


def _export_segment_file(
    segment_samples: NDArray[np.int16],
    *,
    target_path: Path,
    silence: bytes,
    overwrite: bool,
) -> None:
    try:
        atomic_write_file(
            target_path,
            write=lambda temporary_path: _write_wav_file(
                temporary_path,
                segment_samples,
                silence=silence,
            ),
            overwrite=overwrite,
            mode=DEFAULT_FILE_CREATION_MODE,
        )
    except AtomicWriteConflictError:
        raise AudioSegmentExportError(
            EXISTING_SEGMENT_ERROR.format(path=target_path)
        ) from None
    except OSError as error:
        raise AudioSegmentExportError(
            EXPORT_SEGMENT_ERROR.format(
                path=target_path,
                error_message=str(error),
            )
        ) from error


def _build_silence(duration_ms: int) -> bytes:
    """按 16 kHz 单声道 16 位 PCM 格式生成指定时长的静音字节。"""
    sample_count = round(duration_ms * VAD_SAMPLE_RATE / MILLISECONDS_PER_SECOND)
    return bytes(sample_count * PCM_SAMPLE_WIDTH_BYTES)


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


def _format_file_timestamp(timestamp_ms: int) -> str:
    """把毫秒转换为适合放进文件名的“时-分-秒.毫秒”格式。"""
    total_seconds, milliseconds = divmod(timestamp_ms, MILLISECONDS_PER_SECOND)
    total_minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    hours, minutes = divmod(total_minutes, MINUTES_PER_HOUR)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}.{milliseconds:03d}"
