from __future__ import annotations

import subprocess
import tempfile
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio._pcm import (
    PCM_CHANNELS,
    PCM_SAMPLE_WIDTH_BYTES,
)
from podcast_job_finder.filesystem import (
    OWNER_READ_WRITE_MODE,
    temporary_sibling_path,
)


NORMALIZED_AUDIO_FILE_NAME: Final = "normalized.wav"
NORMALIZED_AUDIO_READ_BUFFER_SECONDS: Final = 60
FFMPEG_EXECUTABLE: Final = "ffmpeg"
FFMPEG_LOG_LEVEL: Final = "error"
WAV_AUDIO_CODEC: Final = "pcm_s16le"
WAV_OUTPUT_FORMAT: Final = "wav"
START_FFMPEG_ERROR: Final = "无法启动 ffmpeg：{error_message}"
AUDIO_FILE_NOT_FOUND_ERROR: Final = "音频文件不存在：{path}"
AUDIO_PATH_NOT_FILE_ERROR: Final = "音频路径不是普通文件：{path}"
DECODE_AUDIO_ERROR: Final = "ffmpeg 无法解码音频：{path}，{error_message}"
EMPTY_DECODED_AUDIO_ERROR: Final = "音频解码后没有声音数据：{path}"
INVALID_NORMALIZED_AUDIO_ERROR: Final = "规范化音频格式无效：{path}"
READ_NORMALIZED_AUDIO_ERROR: Final = "无法读取规范化音频：{path}，{error_message}"
NORMALIZED_TEMPORARY_FILE_ERROR: Final = (
    "无法创建或清理规范化音频临时文件：{error_message}"
)


class AudioFileDecodeError(RuntimeError):
    """音频文件无法转换成规范化 WAV 时抛出的错误。"""


@dataclass(slots=True, frozen=True)
class NormalizedAudio:
    """复用同一读取器访问单声道 16 位 PCM WAV。"""

    sample_rate: int
    sample_count: int

    # 由 normalize_audio_file 负责打开和关闭的 WAV 读取器。
    _audio_file: wave.Wave_read = field(repr=False, compare=False)

    def iter_samples(
        self,
        *,
        chunk_samples: int | None = None,
    ) -> Iterator[NDArray[np.int16]]:
        # 每次从 WAV 读取一分钟音频，减少频繁读取文件产生的开销。
        read_samples = self.sample_rate * NORMALIZED_AUDIO_READ_BUFFER_SECONDS
        yielded_samples = chunk_samples or read_samples
        self._audio_file.setpos(0)
        while sample_bytes := self._audio_file.readframes(read_samples):
            samples = np.frombuffer(sample_bytes, dtype=np.int16)
            for start_sample in range(0, samples.size, yielded_samples):
                yield samples[start_sample : start_sample + yielded_samples]

    def read_samples(
        self,
        start_sample: int,
        end_sample: int,
    ) -> NDArray[np.int16]:
        self._audio_file.setpos(start_sample)
        sample_bytes = self._audio_file.readframes(end_sample - start_sample)
        return np.frombuffer(sample_bytes, dtype=np.int16)


@contextmanager
def normalize_audio_file(
    audio_path: Path,
    *,
    sample_rate: int,
) -> Iterator[NormalizedAudio]:
    """用一次 ffmpeg 解码创建临时 WAV，并在上下文结束时清理。"""
    _validate_audio_path(audio_path)
    temporary_target = Path(tempfile.gettempdir()) / NORMALIZED_AUDIO_FILE_NAME
    with temporary_sibling_path(
        temporary_target,
        mode=OWNER_READ_WRITE_MODE,
        error_factory=_build_temporary_file_error,
    ) as normalized_path:
        _decode_to_wav(audio_path, normalized_path, sample_rate=sample_rate)
        with _open_normalized_audio(normalized_path) as audio_file:
            yield _inspect_normalized_audio(
                normalized_path,
                audio_file=audio_file,
                sample_rate=sample_rate,
            )


def _decode_to_wav(
    audio_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
) -> None:
    try:
        completed_process = subprocess.run(
            _build_decode_command(
                audio_path,
                output_path=output_path,
                sample_rate=sample_rate,
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise AudioFileDecodeError(
            START_FFMPEG_ERROR.format(error_message=str(error))
        ) from error

    if completed_process.returncode == 0:
        return
    error_message = completed_process.stderr.decode(errors="replace").strip()
    raise AudioFileDecodeError(
        DECODE_AUDIO_ERROR.format(
            path=audio_path,
            error_message=error_message or completed_process.returncode,
        )
    )


def _validate_audio_path(audio_path: Path) -> None:
    if not audio_path.exists():
        raise AudioFileDecodeError(AUDIO_FILE_NOT_FOUND_ERROR.format(path=audio_path))
    if not audio_path.is_file():
        raise AudioFileDecodeError(AUDIO_PATH_NOT_FILE_ERROR.format(path=audio_path))


def _inspect_normalized_audio(
    audio_path: Path,
    *,
    audio_file: wave.Wave_read,
    sample_rate: int,
) -> NormalizedAudio:
    valid_format = (
        audio_file.getnchannels() == PCM_CHANNELS
        and audio_file.getsampwidth() == PCM_SAMPLE_WIDTH_BYTES
        and audio_file.getframerate() == sample_rate
        and audio_file.getcomptype() == "NONE"
    )
    sample_count = audio_file.getnframes()

    if not valid_format:
        raise AudioFileDecodeError(
            INVALID_NORMALIZED_AUDIO_ERROR.format(path=audio_path)
        )
    if sample_count == 0:
        raise AudioFileDecodeError(EMPTY_DECODED_AUDIO_ERROR.format(path=audio_path))
    return NormalizedAudio(
        sample_rate=sample_rate,
        sample_count=sample_count,
        _audio_file=audio_file,
    )


def _build_temporary_file_error(error_message: str) -> AudioFileDecodeError:
    return AudioFileDecodeError(
        NORMALIZED_TEMPORARY_FILE_ERROR.format(error_message=error_message)
    )


def _open_normalized_audio(audio_path: Path) -> wave.Wave_read:
    try:
        return wave.open(str(audio_path), "rb")
    except (OSError, EOFError, wave.Error) as error:
        raise AudioFileDecodeError(
            READ_NORMALIZED_AUDIO_ERROR.format(
                path=audio_path,
                error_message=str(error),
            )
        ) from error


def _build_decode_command(
    audio_path: Path,
    *,
    output_path: Path,
    sample_rate: int,
) -> list[str]:
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
        str(PCM_CHANNELS),
        "-ar",
        str(sample_rate),
        "-c:a",
        WAV_AUDIO_CODEC,
        "-f",
        WAV_OUTPUT_FORMAT,
        "-y",
        str(output_path),
    ]
