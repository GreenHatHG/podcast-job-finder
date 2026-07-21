from __future__ import annotations

from pathlib import Path

from podcast_job_finder.audio.pcm_decode import AudioFileDecodeError, decode_audio_file
from podcast_job_finder.audio.vad import (
    VAD_SAMPLE_RATE,
    SpeechSegment,
    VadConfig,
    detect_speech_segments,
)


__all__ = ["AudioFileDecodeError", "detect_speech_segments_from_file"]


def detect_speech_segments_from_file(
    audio_path: Path,
    *,
    config: VadConfig = VadConfig(),
) -> list[SpeechSegment]:
    """读取音频文件并返回自然说话片段的开始和结束时间。

    音频会先转换成 VAD 需要的单声道格式和采样率，再交给现有分段流程处理。
    支持的文件类型由本机安装的 ffmpeg 决定。
    """
    samples = decode_audio_file(audio_path, sample_rate=VAD_SAMPLE_RATE)
    return detect_speech_segments(
        samples,
        sample_rate=VAD_SAMPLE_RATE,
        config=config,
    )
