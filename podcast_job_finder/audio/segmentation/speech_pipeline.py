from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from podcast_job_finder.audio.segmentation.normalized_audio import normalize_audio_file
from podcast_job_finder.audio.segmentation.segment_export import (
    DEFAULT_SEGMENT_AUDIO_FORMAT,
    ExportedSpeechSegment,
    SpeechSegmentExportConfig,
    _export_speech_segments,
)
from podcast_job_finder.audio.segmentation.vad import (
    VAD_SAMPLE_RATE,
    VadConfig,
    _detect_speech_segments,
)


DEFAULT_SILENCE_PADDING_MS: Final = 500
INVALID_SILENCE_PADDING_ERROR: Final = "silence_padding_ms 必须大于等于 0。"

logger = logging.getLogger(__name__)


def detect_and_export_speech_segments(
    audio_path: Path,
    *,
    output_dir: Path,
    config: VadConfig = VadConfig(),
    export_config: SpeechSegmentExportConfig = SpeechSegmentExportConfig(
        silence_padding_ms=DEFAULT_SILENCE_PADDING_MS,
        audio_format=DEFAULT_SEGMENT_AUDIO_FORMAT,
    ),
) -> list[ExportedSpeechSegment]:
    """单次规范化解码音频，依次完成 VAD 检测和片段导出。"""
    if export_config.silence_padding_ms < 0:
        raise ValueError(INVALID_SILENCE_PADDING_ERROR)

    logger.info(
        "开始切分音频：audio_path=%s output_dir=%s overwrite=%s",
        audio_path,
        output_dir,
        export_config.overwrite,
    )
    logger.debug(
        "音频切分配置：threshold=%.2f min_speech_duration_ms=%d "
        "max_speech_duration_ms=%d forced_split_overlap_ms=%d "
        "min_silence_duration_ms=%d silence_padding_ms=%d segment_audio_format=%s",
        config.threshold,
        config.min_speech_duration_ms,
        config.max_speech_duration_ms,
        config.forced_split_overlap_ms,
        config.min_silence_duration_ms,
        export_config.silence_padding_ms,
        export_config.audio_format,
    )
    logger.info("开始规范化音频：audio_path=%s", audio_path)
    with normalize_audio_file(audio_path, sample_rate=VAD_SAMPLE_RATE) as audio:
        logger.info(
            "音频规范化完成：duration_seconds=%.1f sample_rate=%d sample_count=%d",
            audio.sample_count / audio.sample_rate,
            audio.sample_rate,
            audio.sample_count,
        )
        logger.info("开始检测语音片段")
        segments = _detect_speech_segments(
            audio,
            config=config,
        )
        logger.info("语音片段检测完成：segment_count=%d", len(segments))
        logger.info(
            "开始导出音频片段：segment_count=%d output_dir=%s",
            len(segments),
            output_dir,
        )
        exported_segments = _export_speech_segments(
            audio,
            segments,
            output_dir,
            export_config,
        )
        logger.info(
            "音频片段导出完成：segment_count=%d output_dir=%s",
            len(exported_segments),
            output_dir,
        )
        return exported_segments
