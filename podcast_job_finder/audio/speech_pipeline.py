from __future__ import annotations

from pathlib import Path
from typing import Final

from podcast_job_finder.audio.normalized_audio import normalize_audio_file
from podcast_job_finder.audio.segment_export import (
    ExportedSpeechSegment,
    _export_speech_segments,
)
from podcast_job_finder.audio.vad import (
    VAD_SAMPLE_RATE,
    VadConfig,
    _detect_speech_segments,
)


DEFAULT_SILENCE_PADDING_MS: Final = 500
INVALID_SILENCE_PADDING_ERROR: Final = "silence_padding_ms 必须大于等于 0。"


def detect_and_export_speech_segments(
    audio_path: Path,
    *,
    output_dir: Path,
    config: VadConfig = VadConfig(),
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
    overwrite: bool = False,
) -> list[ExportedSpeechSegment]:
    """单次规范化解码音频，依次完成 VAD 检测和片段导出。"""
    if silence_padding_ms < 0:
        raise ValueError(INVALID_SILENCE_PADDING_ERROR)
    with normalize_audio_file(audio_path, sample_rate=VAD_SAMPLE_RATE) as audio:
        segments = _detect_speech_segments(
            audio,
            config=config,
        )
        return _export_speech_segments(
            audio, segments, output_dir, silence_padding_ms, overwrite
        )
