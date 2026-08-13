"""Batch podcast audio transcription with resumable segment checkpoints."""

from __future__ import annotations

import logging
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
    RESULT_STATUS_SUCCESS,
    BatchAudioTranscriptionResult,
)
from podcast_job_finder.transcription.manifest import TRANSCRIPTION_FILE_NAME
from podcast_job_finder.audio.episode_audio.service import DEFAULT_AUDIO_OUTPUT_DIR


TRANSCRIPTION_REPORT_TEMPLATE: Final = "transcription_result_{feed_id}_{timestamp}.json"
SAVE_REPORT_ERROR_TEMPLATE: Final = "保存音频转写批次报告失败：{path}，{error_message}"
SEGMENT_DIR_NAME = episode_transcription.SEGMENT_DIR_NAME
MISSING_EPISODE_ID_ERROR = episode_transcription.MISSING_EPISODE_ID_ERROR
MISSING_AUDIO_URL_ERROR = episode_transcription.MISSING_AUDIO_URL_ERROR
EXPECTED_EPISODE_ERRORS = episode_transcription.EXPECTED_EPISODE_ERRORS
RESULT_STATUS_ERROR = pipeline_results.RESULT_STATUS_ERROR

logger = logging.getLogger(__name__)


class BatchAudioTranscriptionError(RuntimeError):
    """批量音频转写流程无法启动或保存批次结果。"""


def run_batch_audio_transcription(  # pylint: disable=too-many-arguments
    *,
    work_items: Sequence[EpisodeWorkItem],
    runtime: AudioTranscriptionRuntime,
    audio_output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
    processing_mode: AudioProcessingMode = DEFAULT_AUDIO_PROCESSING_MODE,
    resume: bool = False,
) -> BatchAudioTranscriptionResult:
    def prepare_episode(
        work_item: EpisodeWorkItem,
    ) -> DownloadedEpisodeAudio | dict[str, object]:
        return prepare_episode_audio(
            work_item=work_item,
            audio_output_dir=audio_output_dir,
            resume=resume,
        )

    def process_episode(
        prepared_episode: DownloadedEpisodeAudio | dict[str, object],
    ) -> dict[str, object]:
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
        1 for result in episode_results if result.get("status") == RESULT_STATUS_SUCCESS
    )
    return BatchAudioTranscriptionResult(
        episode_results=episode_results,
        success_count=success_count,
        fail_count=len(episode_results) - success_count,
    )


def load_existing_batch_transcription_result(
    work_items: Sequence[EpisodeWorkItem],
    *,
    audio_output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
) -> tuple[BatchAudioTranscriptionResult, int]:
    """从已有节目转写清单构造批次结果，并返回缺少清单的节目数量。"""

    episode_results: list[dict[str, object]] = []
    skipped_count = 0
    for work_item in work_items:
        eid = work_item.resolve_episode_id()
        if eid is None:
            skipped_count += 1
            continue
        transcription_path = audio_output_dir / eid / TRANSCRIPTION_FILE_NAME
        if not transcription_path.exists():
            skipped_count += 1
            continue
        record = work_item.to_result_metadata(eid=eid)
        record.update(
            {
                "status": RESULT_STATUS_SUCCESS,
                "cached": True,
                "transcription_path": str(transcription_path),
            }
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
) -> Path:
    timestamp = build_utc_timestamp()
    report_path = output_dir / TRANSCRIPTION_REPORT_TEMPLATE.format(
        feed_id=feed_id,
        timestamp=timestamp.file_label,
    )
    report = {
        "feed_id": feed_id,
        "source": "audio",
        **runtime.metadata,
        "segment_audio_format": runtime.segment_audio_format,
        "created_at": timestamp.text,
        "total": len(result.episode_results),
        "success": result.success_count,
        "failed": result.fail_count,
        "episodes": result.episode_results,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
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
