from __future__ import annotations

from typing import Final


FFMPEG_EXECUTABLE: Final = "ffmpeg"
FFMPEG_LOG_LEVEL: Final = "error"
FFMPEG_AUDIO_CHANNELS: Final = 1
START_FFMPEG_ERROR: Final = "无法启动 ffmpeg：{error_message}"


def build_ffmpeg_command() -> list[str]:
    """返回每次处理音频都需要的 ffmpeg 基础命令。"""
    return [
        FFMPEG_EXECUTABLE,
        "-hide_banner",
        "-loglevel",
        FFMPEG_LOG_LEVEL,
        "-nostdin",
    ]


def build_audio_output_arguments(*, sample_rate: int) -> list[str]:
    """返回选择第一条音轨并统一声道和采样率所需的参数。"""
    return [
        "-map",
        "0:a:0",
        "-ac",
        str(FFMPEG_AUDIO_CHANNELS),
        "-ar",
        str(sample_rate),
    ]
