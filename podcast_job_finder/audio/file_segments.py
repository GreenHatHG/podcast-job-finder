from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.vad import (
    VAD_SAMPLE_RATE,
    SpeechSegment,
    VadConfig,
    detect_speech_segments,
)


FFMPEG_EXECUTABLE: Final = "ffmpeg"
FFMPEG_LOG_LEVEL: Final = "error"
FFMPEG_AUDIO_CHANNELS: Final = 1
FFMPEG_SAMPLE_FORMAT: Final = "s16le"
FFMPEG_PIPE_OUTPUT: Final = "pipe:1"
AUDIO_FILE_NOT_FOUND_ERROR: Final = "音频文件不存在：{path}"
AUDIO_PATH_NOT_FILE_ERROR: Final = "音频路径不是普通文件：{path}"
START_FFMPEG_ERROR: Final = "无法启动 ffmpeg：{error_message}"
DECODE_AUDIO_ERROR: Final = "ffmpeg 无法解码音频：{path}，{error_message}"
EMPTY_DECODED_AUDIO_ERROR: Final = "音频解码后没有声音数据：{path}"


class AudioFileDecodeError(RuntimeError):
    """音频文件无法转换成 VAD 所需格式时抛出的错误。"""


def detect_speech_segments_from_file(
    audio_path: Path,
    *,
    config: VadConfig = VadConfig(),
) -> list[SpeechSegment]:
    """读取音频文件并返回自然说话片段的开始和结束时间。

    音频会先转换成 VAD 需要的单声道格式和采样率，再交给现有分段流程处理。
    支持的文件类型由本机安装的 ffmpeg 决定。
    """
    samples = _decode_audio_file(audio_path)
    return detect_speech_segments(
        samples,
        sample_rate=VAD_SAMPLE_RATE,
        config=config,
    )


def _decode_audio_file(audio_path: Path) -> NDArray[np.int16]:
    """使用 ffmpeg 读取音频，并转换成 VAD 可以直接处理的声音数据。"""
    _validate_audio_path(audio_path)
    command = _build_decode_command(audio_path)
    try:
        completed_process = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise AudioFileDecodeError(
            START_FFMPEG_ERROR.format(error_message=str(error))
        ) from error

    if completed_process.returncode != 0:
        error_message = completed_process.stderr.decode(errors="replace").strip()
        raise AudioFileDecodeError(
            DECODE_AUDIO_ERROR.format(
                path=audio_path,
                error_message=error_message or completed_process.returncode,
            )
        )
    if not completed_process.stdout:
        raise AudioFileDecodeError(EMPTY_DECODED_AUDIO_ERROR.format(path=audio_path))

    samples: NDArray[np.int16] = np.frombuffer(
        completed_process.stdout,
        dtype=np.int16,
    )
    return samples


def _validate_audio_path(audio_path: Path) -> None:
    if not audio_path.exists():
        raise AudioFileDecodeError(AUDIO_FILE_NOT_FOUND_ERROR.format(path=audio_path))
    if not audio_path.is_file():
        raise AudioFileDecodeError(AUDIO_PATH_NOT_FILE_ERROR.format(path=audio_path))


def _build_decode_command(audio_path: Path) -> list[str]:
    return [
        FFMPEG_EXECUTABLE,
        "-hide_banner",
        "-loglevel",
        FFMPEG_LOG_LEVEL,
        "-nostdin",
        "-i",
        str(audio_path),
        "-map",
        "0:a:0",
        "-ac",
        str(FFMPEG_AUDIO_CHANNELS),
        "-ar",
        str(VAD_SAMPLE_RATE),
        "-f",
        FFMPEG_SAMPLE_FORMAT,
        FFMPEG_PIPE_OUTPUT,
    ]
