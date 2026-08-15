from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Final

from podcast_job_finder.audio.segmentation.normalized_audio import normalize_audio_file
from podcast_job_finder.audio.segmentation.segment_export import (
    DEFAULT_SEGMENT_AUDIO_FORMAT,
    ExportedSpeechSegment,
    SegmentAudioFormat,
    _export_speech_segments,
)
from podcast_job_finder.audio.segmentation.vad import (
    VAD_SAMPLE_RATE,
    SpeechSegment,
    VadConfig,
    _detect_speech_segments,
)
from podcast_job_finder.timestamps import format_duration_ms


DEFAULT_SILENCE_PADDING_MS: Final = 500
INVALID_SILENCE_PADDING_ERROR: Final = "silence_padding_ms 必须大于等于 0。"
DURATION_BUCKET_BOUNDARY_MS: Final = 10_000

logger = logging.getLogger(__name__)


def detect_and_export_speech_segments(  # pylint: disable=too-many-arguments
    audio_path: Path,
    *,
    output_dir: Path,
    config: VadConfig = VadConfig(),
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
    audio_format: SegmentAudioFormat = DEFAULT_SEGMENT_AUDIO_FORMAT,
    overwrite: bool = False,
) -> list[ExportedSpeechSegment]:
    """单次规范化解码音频，依次完成 VAD 检测和片段导出。"""
    if silence_padding_ms < 0:
        raise ValueError(INVALID_SILENCE_PADDING_ERROR)

    logger.info(
        "开始切分音频：audio_path=%s output_dir=%s overwrite=%s",
        audio_path,
        output_dir,
        overwrite,
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
        silence_padding_ms,
        audio_format,
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
        _log_speech_segment_summary(segments)
        logger.info(
            "开始导出音频片段：segment_count=%d output_dir=%s",
            len(segments),
            output_dir,
        )
        exported_segments = _export_speech_segments(
            audio,
            segments,
            output_dir,
            silence_padding_ms=silence_padding_ms,
            audio_format=audio_format,
            overwrite=overwrite,
        )
        logger.info(
            "音频片段导出完成：segment_count=%d output_dir=%s",
            len(exported_segments),
            output_dir,
        )
        return exported_segments


def _log_speech_segment_summary(segments: Sequence[SpeechSegment]) -> None:
    if not segments:
        logger.info("语音片段检测完成：segment_count=0 duration_distribution=无片段")
        return

    durations_ms = [segment.duration_ms for segment in segments]
    logger.info(
        "语音片段检测完成：segment_count=%d total_duration=%s "
        "min_duration=%s max_duration=%s average_duration=%s "
        'median_duration=%s duration_distribution="%s"',
        len(durations_ms),
        format_duration_ms(sum(durations_ms)),
        format_duration_ms(min(durations_ms)),
        format_duration_ms(max(durations_ms)),
        format_duration_ms(round(sum(durations_ms) / len(durations_ms))),
        format_duration_ms(round(median(durations_ms))),
        _format_duration_distribution(durations_ms),
    )


def _format_duration_distribution(durations_ms: Sequence[int]) -> str:
    under_10_seconds = 0
    from_10_to_20_seconds = 0
    from_20_to_30_seconds = 0
    at_least_30_seconds = 0
    for duration_ms in durations_ms:
        if duration_ms < DURATION_BUCKET_BOUNDARY_MS:
            under_10_seconds += 1
        elif duration_ms < DURATION_BUCKET_BOUNDARY_MS * 2:
            from_10_to_20_seconds += 1
        elif duration_ms < DURATION_BUCKET_BOUNDARY_MS * 3:
            from_20_to_30_seconds += 1
        else:
            at_least_30_seconds += 1

    total_count = len(durations_ms)
    buckets = (
        ("<10s", under_10_seconds),
        ("10-20s", from_10_to_20_seconds),
        ("20-30s", from_20_to_30_seconds),
        (">=30s", at_least_30_seconds),
    )
    return ", ".join(
        f"{label}={count} ({count / total_count:.1%})" for label, count in buckets
    )
