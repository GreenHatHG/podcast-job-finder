"""Batch podcast audio transcription with resumable segment checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    ExportedSpeechSegment,
)
from podcast_job_finder.transcription.schedule import (
    DEFAULT_AUDIO_PROCESSING_MODE,
    AudioProcessingMode,
    run_audio_processing_schedule,
)
from podcast_job_finder.audio.segmentation.speech_segment_checkpoint import (
    SPEECH_SEGMENT_CHECKPOINT_FILE_NAME,
    restore_or_export_speech_segments,
)
from podcast_job_finder.transcription.models import (
    AudioTranscriptionError,
    AudioTranscriptionResult,
)
from podcast_job_finder.transcription.completed_transcription import (
    can_restore_completed_transcription,
)
from podcast_job_finder.transcription.runtime import AudioTranscriptionRuntime
from podcast_job_finder.transcription.formatting.article import (
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    save_transcription_article,
)
from podcast_job_finder.transcription.manifest import (
    TRANSCRIPTION_FILE_NAME,
    save_audio_transcription_manifest,
)
from podcast_job_finder.transcription.quality_report import (
    TRANSCRIPTION_QUALITY_REPORT_FILE_NAME,
    save_transcription_quality_report,
)
from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_json,
)
from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioDownloadError
from podcast_job_finder.audio.episode_audio.files import (
    build_audio_target_path,
    prepare_episode_audio_directory,
    store_episode_audio,
)
from podcast_job_finder.audio.episode_audio.service import (
    DEFAULT_AUDIO_OUTPUT_DIR,
    extract_audio_extension,
)
from podcast_job_finder.transcription.checkpoint import (
    SegmentTranscriptionCheckpointStore,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.timestamps import build_utc_timestamp


SEGMENT_DIR_NAME: Final = "segments"
TRANSCRIPTION_REPORT_TEMPLATE: Final = "transcription_result_{feed_id}_{timestamp}.json"
RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"
MISSING_EPISODE_ID_ERROR: Final = "音频转写任务缺少有效的节目 ID：{url}"
MISSING_AUDIO_URL_ERROR: Final = "RSS 节目缺少音频地址：eid={eid}"
SAVE_REPORT_ERROR_TEMPLATE: Final = "保存音频转写批次报告失败：{path}，{error_message}"

logger = logging.getLogger(__name__)

EXPECTED_EPISODE_ERRORS = (
    AudioFileDecodeError,
    AudioSegmentExportError,
    AudioTranscriptionError,
    EpisodeAudioDownloadError,
    OSError,
    ValueError,
)


class BatchAudioTranscriptionError(RuntimeError):
    """批量音频转写流程无法启动或保存批次结果。"""


@dataclass(slots=True, frozen=True)
class BatchAudioTranscriptionResult:
    episode_results: list[dict[str, object]]
    success_count: int
    fail_count: int


@dataclass(slots=True, frozen=True)
class _EpisodeTranscriptionContext:
    work_item: EpisodeWorkItem
    eid: str
    transcription_path: Path

    @property
    def article_path(self) -> Path:
        return self.transcription_path.with_name(TRANSCRIPTION_ARTICLE_FILE_NAME)

    @property
    def quality_report_path(self) -> Path:
        return self.transcription_path.with_name(TRANSCRIPTION_QUALITY_REPORT_FILE_NAME)

    @property
    def segment_checkpoint_path(self) -> Path:
        return self.transcription_path.with_name(SPEECH_SEGMENT_CHECKPOINT_FILE_NAME)

    @property
    def segment_dir(self) -> Path:
        return self.transcription_path.parent / SEGMENT_DIR_NAME


@dataclass(slots=True, frozen=True)
class _DownloadedEpisodeAudio:
    context: _EpisodeTranscriptionContext
    local_path: Path
    source_url: str


def _build_segment_checkpoint_store(
    context: _EpisodeTranscriptionContext,
) -> SegmentTranscriptionCheckpointStore:
    return SegmentTranscriptionCheckpointStore(
        metadata={
            "eid": context.eid,
            "episode_url": context.work_item.episode_url,
            "title": context.work_item.title,
            "pub_date": context.work_item.pub_date,
        },
        expected_metadata={
            "eid": context.eid,
            "episode_url": context.work_item.episode_url,
        },
    )


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
    ) -> _DownloadedEpisodeAudio | dict[str, object]:
        return _prepare_episode_audio(
            work_item=work_item,
            audio_output_dir=audio_output_dir,
            resume=resume,
        )

    def process_episode(
        prepared_episode: _DownloadedEpisodeAudio | dict[str, object],
    ) -> dict[str, object]:
        return _transcribe_prepared_episode(
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


def _prepare_episode_audio(
    *,
    work_item: EpisodeWorkItem,
    audio_output_dir: Path,
    resume: bool,
) -> _DownloadedEpisodeAudio | dict[str, object]:
    try:
        context = _build_episode_context(
            work_item=work_item,
            audio_output_dir=audio_output_dir,
        )
        if resume and can_restore_completed_transcription(
            transcription_path=context.transcription_path,
            article_path=context.article_path,
            quality_report_path=context.quality_report_path,
            segment_dir=context.segment_dir,
            article_title=context.work_item.title or context.eid,
            expected_metadata={
                "eid": context.eid,
                "episode_url": context.work_item.episode_url,
            },
        ):
            logger.info("命中完整音频转写清单：eid=%s", context.eid)
            return _build_success_record(context, cached=True)

        logger.info("下载节目音频：eid=%s title=%s", context.eid, work_item.title)
        source_url = work_item.audio_url
        if not source_url:
            raise EpisodeAudioDownloadError(
                MISSING_AUDIO_URL_ERROR.format(eid=context.eid)
            )
        extension = extract_audio_extension(source_url)
        local_path = build_audio_target_path(audio_output_dir, context.eid, extension)
        store_episode_audio(source_url, local_path, overwrite=False)
        return _DownloadedEpisodeAudio(
            context=context,
            local_path=local_path,
            source_url=source_url,
        )
    except EXPECTED_EPISODE_ERRORS as error:
        logger.info("节目音频准备失败：%s", error)
        return _build_error_record(work_item, str(error))


def _build_episode_context(
    *,
    work_item: EpisodeWorkItem,
    audio_output_dir: Path,
) -> _EpisodeTranscriptionContext:
    eid = work_item.resolve_episode_id()
    if eid is None:
        raise ValueError(MISSING_EPISODE_ID_ERROR.format(url=work_item.episode_url))
    episode_output_dir = prepare_episode_audio_directory(audio_output_dir, eid)
    return _EpisodeTranscriptionContext(
        work_item=work_item,
        eid=eid,
        transcription_path=episode_output_dir / TRANSCRIPTION_FILE_NAME,
    )


def _transcribe_prepared_episode(
    prepared_episode: _DownloadedEpisodeAudio | dict[str, object],
    *,
    runtime: AudioTranscriptionRuntime,
    resume: bool,
) -> dict[str, object]:
    if isinstance(prepared_episode, dict):
        return prepared_episode

    context = prepared_episode.context
    try:
        logger.info(
            "转写节目音频：eid=%s title=%s", context.eid, context.work_item.title
        )
        checkpoint_store = _build_segment_checkpoint_store(context)
        exported_segments = restore_or_export_speech_segments(
            source_path=prepared_episode.local_path,
            output_dir=context.segment_dir,
            checkpoint_path=context.segment_checkpoint_path,
            vad_config=runtime.vad_config,
            silence_padding_ms=runtime.silence_padding_ms,
            audio_format=runtime.segment_audio_format,
            overwrite=True,
            resume=resume,
        )
        transcription_result, all_segments_cached = (
            transcribe_speech_segments_with_checkpoints(
                exported_segments,
                transcriber=runtime.transcriber,
                checkpoint_store=checkpoint_store,
                resume=resume,
            )
        )
        _save_episode_transcription(
            prepared_episode,
            runtime=runtime,
            result=transcription_result,
            exported_segments=exported_segments,
        )
        return _build_success_record(
            context,
            cached=all_segments_cached,
        )
    except EXPECTED_EPISODE_ERRORS as error:
        logger.info("节目音频转写失败：%s", error)
        return _build_error_record(context.work_item, str(error))


def _save_episode_transcription(  # pylint: disable=too-many-arguments
    prepared_episode: _DownloadedEpisodeAudio,
    *,
    runtime: AudioTranscriptionRuntime,
    result: AudioTranscriptionResult,
    exported_segments: Sequence[ExportedSpeechSegment],
) -> None:
    context = prepared_episode.context
    save_transcription_article(
        context.article_path,
        title=context.work_item.title or context.eid,
        body=result.text,
    )
    save_transcription_quality_report(
        context.quality_report_path,
        result,
        exported_segments=exported_segments,
    )
    save_audio_transcription_manifest(
        context.transcription_path,
        metadata={
            "eid": context.eid,
            "title": context.work_item.title,
            "pub_date": context.work_item.pub_date,
            "episode_url": context.work_item.episode_url,
            **runtime.metadata,
            "segment_audio_format": runtime.segment_audio_format,
            "audio_path": str(prepared_episode.local_path),
            "source_url": prepared_episode.source_url,
            "article_path": str(context.article_path),
            "transcription_quality_report_path": str(context.quality_report_path),
        },
        exported_segments=exported_segments,
        result=result,
    )


def _build_success_record(
    context: _EpisodeTranscriptionContext,
    *,
    cached: bool,
) -> dict[str, object]:
    record = context.work_item.to_result_metadata(eid=context.eid)
    record.update(
        {
            "status": RESULT_STATUS_SUCCESS,
            "cached": cached,
            "transcription_path": str(context.transcription_path),
            "article_path": str(context.article_path),
            "transcription_quality_report_path": str(context.quality_report_path),
            "segment_directory": str(context.segment_dir),
        }
    )
    return record


def _build_error_record(
    work_item: EpisodeWorkItem,
    error_message: str,
) -> dict[str, object]:
    record = work_item.to_result_metadata()
    record.update(
        {
            "status": RESULT_STATUS_ERROR,
            "cached": False,
            "error": error_message,
        }
    )
    return record
