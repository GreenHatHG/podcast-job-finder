from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.audio._ffmpeg import (
    START_FFMPEG_ERROR,
    build_audio_output_arguments,
    build_ffmpeg_command,
)
from podcast_job_finder.audio.vad import VAD_SAMPLE_RATE, SpeechSegment


SEGMENT_FILE_NAME_TEMPLATE: Final = "segment_{index:04d}_{start_time}_{end_time}.wav"
PARTIAL_FILE_SUFFIX: Final = ".part"
DEFAULT_SILENCE_PADDING_MS: Final = 500
MILLISECONDS_PER_SECOND: Final = 1_000
SECONDS_PER_MINUTE: Final = 60
MINUTES_PER_HOUR: Final = 60
WAV_OUTPUT_FORMAT: Final = "wav"
WAV_AUDIO_CODEC: Final = "pcm_s16le"
INVALID_SILENCE_PADDING_ERROR: Final = "silence_padding_ms 必须大于等于 0。"
INVALID_SEGMENT_ERROR: Final = "音频片段时间无效：start_ms={start_ms}，end_ms={end_ms}"
OUTPUT_DIR_ERROR: Final = "无法创建音频片段目录：{path}，{error_message}"
EXISTING_SEGMENT_ERROR: Final = "音频片段文件已经存在：{path}"
EXPORT_SEGMENT_ERROR: Final = "ffmpeg 无法导出音频片段：{path}，{error_message}"
EMPTY_SEGMENT_ERROR: Final = "ffmpeg 导出的音频片段为空：{path}"
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

    exported_segments = []
    for index, segment in enumerate(segments, start=1):
        _validate_segment(segment)
        target_path = output_dir / SEGMENT_FILE_NAME_TEMPLATE.format(
            index=index,
            start_time=_format_file_timestamp(segment.start_ms),
            end_time=_format_file_timestamp(segment.end_ms),
        )
        _export_segment_file(
            audio_path,
            segment,
            target_path=target_path,
            silence_padding_ms=silence_padding_ms,
            overwrite=overwrite,
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
    audio_path: Path,
    segment: SpeechSegment,
    *,
    target_path: Path,
    silence_padding_ms: int,
    overwrite: bool,
) -> None:
    if target_path.exists() and not overwrite:
        raise AudioSegmentExportError(EXISTING_SEGMENT_ERROR.format(path=target_path))

    partial_path = target_path.with_suffix(f"{target_path.suffix}{PARTIAL_FILE_SUFFIX}")
    partial_path.unlink(missing_ok=True)
    command = _build_export_command(
        audio_path,
        segment,
        output_path=partial_path,
        silence_padding_ms=silence_padding_ms,
    )
    try:
        _run_export_command(command, partial_path)
        _publish_segment_file(partial_path, target_path)
    finally:
        partial_path.unlink(missing_ok=True)


def _run_export_command(command: list[str], output_path: Path) -> None:
    try:
        completed_process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise AudioSegmentExportError(
            START_FFMPEG_ERROR.format(error_message=str(error))
        ) from error

    if completed_process.returncode != 0:
        error_message = completed_process.stderr.decode(errors="replace").strip()
        raise AudioSegmentExportError(
            EXPORT_SEGMENT_ERROR.format(
                path=output_path,
                error_message=error_message or completed_process.returncode,
            )
        )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise AudioSegmentExportError(EMPTY_SEGMENT_ERROR.format(path=output_path))


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


def _build_export_command(
    audio_path: Path,
    segment: SpeechSegment,
    *,
    output_path: Path,
    silence_padding_ms: int,
) -> list[str]:
    command = [
        *build_ffmpeg_command(),
        "-ss",
        _format_milliseconds(segment.start_ms),
        "-t",
        _format_milliseconds(segment.duration_ms),
        "-i",
        str(audio_path),
        *build_audio_output_arguments(sample_rate=VAD_SAMPLE_RATE),
    ]
    if silence_padding_ms:
        command.extend(["-af", _build_padding_filter(silence_padding_ms)])
    command.extend(
        [
            "-c:a",
            WAV_AUDIO_CODEC,
            "-f",
            WAV_OUTPUT_FORMAT,
            "-y",
            str(output_path),
        ]
    )
    return command


def _build_padding_filter(silence_padding_ms: int) -> str:
    padding_seconds = silence_padding_ms / MILLISECONDS_PER_SECOND
    return f"adelay={silence_padding_ms}:all=1,apad=pad_dur={padding_seconds}"


def _format_milliseconds(duration_ms: int) -> str:
    return f"{duration_ms / MILLISECONDS_PER_SECOND:.3f}"


def _format_file_timestamp(timestamp_ms: int) -> str:
    """把毫秒转换为适合放进文件名的“时-分-秒.毫秒”格式。"""
    total_seconds, milliseconds = divmod(timestamp_ms, MILLISECONDS_PER_SECOND)
    total_minutes, seconds = divmod(total_seconds, SECONDS_PER_MINUTE)
    hours, minutes = divmod(total_minutes, MINUTES_PER_HOUR)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}.{milliseconds:03d}"
