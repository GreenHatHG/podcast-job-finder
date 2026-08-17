"""Batch podcast audio transcription with resumable segment checkpoints."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.transcription.schedule import (
    DEFAULT_AUDIO_PROCESSING_MODE,
    AudioProcessingMode,
    run_audio_processing_schedule,
)
from podcast_job_finder.transcription.runtime import AudioTranscriptionRuntime
from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_json,
)
from podcast_job_finder.timestamps import build_utc_timestamp
from podcast_job_finder.transcription import episode_transcription
from podcast_job_finder.transcription.episode_transcription import (
    DownloadedEpisodeAudio,
    prepare_episode_audio,
    transcribe_prepared_episode,
)
from podcast_job_finder.transcription import pipeline_results
from podcast_job_finder.transcription.pipeline_results import (
    EpisodeTranscriptionResult,
    BatchAudioTranscriptionResult,
    SuccessfulEpisodeTranscriptionResult,
)
from podcast_job_finder.transcription.manifest import TRANSCRIPTION_FILE_NAME
from podcast_job_finder.errors import PodcastJobFinderError
from podcast_job_finder.output_paths import (
    EPISODE_OUTPUT_DIR,
    EPISODE_TRANSCRIPTION_DIR_NAME,
    TRANSCRIPTION_REPORT_DIR_NAME,
    find_episode_output_dir,
    build_feed_report_dir,
)


TRANSCRIPTION_REPORT_TEMPLATE: Final = "{timestamp}.json"
SAVE_REPORT_ERROR_TEMPLATE: Final = "保存音频转写批次报告失败：{path}，{error_message}"
SEGMENT_DIR_NAME = episode_transcription.SEGMENT_DIR_NAME
MISSING_EPISODE_ID_ERROR = episode_transcription.MISSING_EPISODE_ID_ERROR
MISSING_AUDIO_URL_ERROR = episode_transcription.MISSING_AUDIO_URL_ERROR
RESULT_STATUS_ERROR = pipeline_results.RESULT_STATUS_ERROR

logger = logging.getLogger(__name__)


class BatchAudioTranscriptionError(PodcastJobFinderError, RuntimeError):
    """批量音频转写流程无法启动或保存批次结果。"""


def run_batch_audio_transcription(  # pylint: disable=too-many-arguments
    *,
    work_items: Sequence[EpisodeWorkItem],
    runtime: AudioTranscriptionRuntime,
    audio_output_dir: Path = EPISODE_OUTPUT_DIR,
    processing_mode: AudioProcessingMode = DEFAULT_AUDIO_PROCESSING_MODE,
    resume: bool = False,
) -> BatchAudioTranscriptionResult:
    def prepare_episode(
        work_item: EpisodeWorkItem,
    ) -> DownloadedEpisodeAudio | EpisodeTranscriptionResult:
        return prepare_episode_audio(
            work_item=work_item,
            audio_output_dir=audio_output_dir,
            resume=resume,
        )

    def process_episode(
        prepared_episode: DownloadedEpisodeAudio | EpisodeTranscriptionResult,
    ) -> EpisodeTranscriptionResult:
        return transcribe_prepared_episode(
            prepared_episode,
            runtime=runtime,
            resume=resume,
        )

    episode_results = run_audio_processing_schedule(
        work_items=work_items,
        processing_mode=processing_mode,
        prepare_episode=prepare_episode,
        process_episode=process_episode,
    )
    success_count = sum(
        isinstance(result, SuccessfulEpisodeTranscriptionResult)
        for result in episode_results
    )
    return BatchAudioTranscriptionResult(
        episode_results=episode_results,
        success_count=success_count,
        fail_count=len(episode_results) - success_count,
    )


def load_existing_batch_transcription_result(
    work_items: Sequence[EpisodeWorkItem],
    *,
    audio_output_dir: Path = EPISODE_OUTPUT_DIR,
) -> tuple[BatchAudioTranscriptionResult, int]:
    """从已有节目转写清单构造批次结果，并返回缺少清单的节目数量。"""

    episode_results: list[EpisodeTranscriptionResult] = []
    skipped_count = 0
    for work_item in work_items:
        eid = work_item.resolve_episode_id()
        if eid is None:
            logger.debug(
                "跳过已有音频转写：原因=无法解析节目 ID title=%s episode_url=%s",
                work_item.title,
                work_item.episode_url,
            )
            skipped_count += 1
            continue
        episode_output_dir = find_episode_output_dir(
            audio_output_dir,
            eid,
            podcast_title=work_item.podcast_title,
            episode_title=work_item.title,
        )
        transcription_path = (
            episode_output_dir
            / EPISODE_TRANSCRIPTION_DIR_NAME
            / TRANSCRIPTION_FILE_NAME
        )
        if not transcription_path.exists():
            logger.debug(
                "跳过已有音频转写：原因=缺少转写清单 "
                "episode_id=%s title=%s transcription_path=%s",
                eid,
                work_item.title,
                transcription_path,
            )
            skipped_count += 1
            continue
        record = SuccessfulEpisodeTranscriptionResult(
            episode=replace(work_item, eid=eid),
            cached=True,
            episode_output_dir=str(episode_output_dir),
            transcription_path=str(transcription_path),
        )
        episode_results.append(record)

    return (
        BatchAudioTranscriptionResult(
            episode_results=episode_results,
            success_count=len(episode_results),
            fail_count=0,
        ),
        skipped_count,
    )


def save_batch_audio_transcription_report(
    *,
    feed_id: str,
    runtime: AudioTranscriptionRuntime,
    result: BatchAudioTranscriptionResult,
    output_dir: Path,
    podcast_title: str = "podcast",
) -> Path:
    timestamp = build_utc_timestamp()
    report_dir = build_feed_report_dir(
        feed_id,
        TRANSCRIPTION_REPORT_DIR_NAME,
        podcast_title=podcast_title,
        output_dir=output_dir,
    )
    report_path = report_dir / TRANSCRIPTION_REPORT_TEMPLATE.format(
        timestamp=timestamp.file_label
    )
    report = {
        "feed_id": feed_id,
        "podcast_title": podcast_title,
        "source": "audio",
        **runtime.metadata,
        "segment_audio_format": runtime.segment_audio_format,
        "created_at": timestamp.text,
        "total": len(result.episode_results),
        "success": result.success_count,
        "failed": result.fail_count,
        "episodes": [episode.to_dict() for episode in result.episode_results],
    }
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            report_path,
            report,
            mode=DEFAULT_FILE_CREATION_MODE,
        )
    except OSError as error:
        raise BatchAudioTranscriptionError(
            SAVE_REPORT_ERROR_TEMPLATE.format(
                path=report_path,
                error_message=str(error),
            )
        ) from error
    return report_path
