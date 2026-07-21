from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio._ffmpeg import (
    START_FFMPEG_ERROR,
    build_audio_output_arguments,
    build_ffmpeg_command,
)


FFMPEG_SAMPLE_FORMAT: Final = "s16le"
FFMPEG_PIPE_OUTPUT: Final = "pipe:1"
AUDIO_FILE_NOT_FOUND_ERROR: Final = "音频文件不存在：{path}"
AUDIO_PATH_NOT_FILE_ERROR: Final = "音频路径不是普通文件：{path}"
DECODE_AUDIO_ERROR: Final = "ffmpeg 无法解码音频：{path}，{error_message}"
EMPTY_DECODED_AUDIO_ERROR: Final = "音频解码后没有声音数据：{path}"


class AudioFileDecodeError(RuntimeError):
    """音频文件无法转换成单声道 16 位 PCM 时抛出的错误。"""


def decode_audio_file(
    audio_path: Path,
    *,
    sample_rate: int,
) -> NDArray[np.int16]:
    """使用一次 ffmpeg 调用把音频解码成指定采样率的单声道 PCM。"""
    _validate_audio_path(audio_path)
    try:
        completed_process = subprocess.run(
            _build_decode_command(audio_path, sample_rate=sample_rate),
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


def _build_decode_command(audio_path: Path, *, sample_rate: int) -> list[str]:
    return [
        *build_ffmpeg_command(),
        "-i",
        str(audio_path),
        *build_audio_output_arguments(sample_rate=sample_rate),
        "-f",
        FFMPEG_SAMPLE_FORMAT,
        FFMPEG_PIPE_OUTPUT,
    ]
