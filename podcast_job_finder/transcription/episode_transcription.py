"""单个节目音频的下载、切分、转写和结果保存。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.audio import ExportedSpeechSegment
from podcast_job_finder.audio.episode_audio.errors import (
    EpisodeAudioDownloadError,
    EpisodeAudioNotFoundError,
)
from podcast_job_finder.audio.episode_audio.files import (
    build_audio_target_path,
    prepare_episode_output_directory,
    store_episode_audio,
)
from podcast_job_finder.audio.episode_audio.service import extract_audio_extension
from podcast_job_finder.audio.segmentation.speech_segment_checkpoint import (
    SPEECH_SEGMENT_CHECKPOINT_FILE_NAME,
    restore_or_export_speech_segments,
)
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.transcription.checkpoint import (
    SegmentTranscriptionCheckpointStore,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.transcription.completed_transcription import (
    can_restore_completed_transcription,
)
from podcast_job_finder.transcription.formatting.article import (
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    save_transcription_article,
)
from podcast_job_finder.transcription.manifest import (
    TRANSCRIPTION_FILE_NAME,
    save_audio_transcription_manifest,
)
from podcast_job_finder.transcription.models import AudioTranscriptionResult
from podcast_job_finder.transcription.pipeline_results import (
    EpisodeTranscriptionResult,
    FailedEpisodeTranscriptionResult,
    SkippedEpisodeTranscriptionResult,
    SuccessfulEpisodeTranscriptionResult,
)
from podcast_job_finder.transcription.quality_report import (
    TRANSCRIPTION_QUALITY_REPORT_FILE_NAME,
    save_transcription_quality_report,
)
from podcast_job_finder.transcription.runtime import AudioTranscriptionRuntime
from podcast_job_finder.errors import EpisodeProcessingError
from podcast_job_finder.output_paths import (
    EPISODE_OUTPUT_DIR,
    EPISODE_AUDIO_DIR_NAME,
    EPISODE_TRANSCRIPTION_DIR_NAME,
)


SEGMENT_DIR_NAME: Final = "segments"
MISSING_EPISODE_ID_ERROR: Final = "音频转写任务缺少有效的节目 ID：{url}"
MISSING_AUDIO_URL_ERROR: Final = "RSS 节目缺少音频地址：eid={eid}"

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _EpisodeTranscriptionContext:
    work_item: EpisodeWorkItem
    eid: str
    episode_dir: Path

    @property
    def transcription_path(self) -> Path:
        return (
            self.episode_dir / EPISODE_TRANSCRIPTION_DIR_NAME / TRANSCRIPTION_FILE_NAME
        )

    @property
    def article_path(self) -> Path:
        return self.transcription_path.with_name(TRANSCRIPTION_ARTICLE_FILE_NAME)

    @property
    def quality_report_path(self) -> Path:
        return self.transcription_path.with_name(TRANSCRIPTION_QUALITY_REPORT_FILE_NAME)

    @property
    def segment_checkpoint_path(self) -> Path:
        return (
            self.episode_dir
            / EPISODE_AUDIO_DIR_NAME
            / SPEECH_SEGMENT_CHECKPOINT_FILE_NAME
        )

    @property
    def segment_dir(self) -> Path:
        return self.episode_dir / EPISODE_AUDIO_DIR_NAME / SEGMENT_DIR_NAME


@dataclass(slots=True, frozen=True)
class DownloadedEpisodeAudio:
    context: _EpisodeTranscriptionContext
    local_path: Path
    source_url: str


def prepare_episode_audio(
    *,
    work_item: EpisodeWorkItem,
    audio_output_dir: Path = EPISODE_OUTPUT_DIR,
    resume: bool,
) -> DownloadedEpisodeAudio | EpisodeTranscriptionResult:
    """准备单个节目的本地音频，命中完整清单时直接返回成功记录。"""

    try:
        context = _build_episode_context(
            work_item=work_item,
            audio_output_dir=audio_output_dir,
        )
        if resume and can_restore_completed_transcription(
            transcription_path=context.transcription_path,
            article_path=context.article_path,
            quality_report_path=context.quality_report_path,
            article_title=context.work_item.title or context.eid,
            expected_metadata={
                "eid": context.eid,
                "episode_url": context.work_item.episode_url,
            },
        ):
            logger.info("命中完整音频转写清单：eid=%s", context.eid)
            return _build_successful_episode_result(context, cached=True)

        logger.info("下载节目音频：eid=%s title=%s", context.eid, work_item.title)
        source_url = work_item.audio_url
        if not source_url:
            raise EpisodeAudioDownloadError(
                MISSING_AUDIO_URL_ERROR.format(eid=context.eid)
            )
        extension = extract_audio_extension(source_url)
        local_path = build_audio_target_path(
            audio_output_dir,
            context.eid,
            extension,
            podcast_title=work_item.podcast_title,
            episode_title=work_item.title,
        )
        store_episode_audio(source_url, local_path, overwrite=False)
        return DownloadedEpisodeAudio(
            context=context,
            local_path=local_path,
            source_url=source_url,
        )
    except EpisodeAudioNotFoundError as error:
        logger.warning(
            "节目音频 404，重试后跳过：eid=%s error=%s",
            work_item.resolve_episode_id() or "unknown",
            error,
        )
        return SkippedEpisodeTranscriptionResult(
            episode=work_item,
            reason=str(error),
        )
    except EpisodeProcessingError as error:
        logger.info("节目音频准备失败：%s", error)
        return FailedEpisodeTranscriptionResult(
            episode=work_item,
            error=str(error),
        )


def transcribe_prepared_episode(
    prepared_episode: DownloadedEpisodeAudio | EpisodeTranscriptionResult,
    *,
    runtime: AudioTranscriptionRuntime,
    resume: bool,
) -> EpisodeTranscriptionResult:
    """切分并转写已准备好的单个节目音频。"""

    if not isinstance(prepared_episode, DownloadedEpisodeAudio):
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
        return _build_successful_episode_result(
            context,
            cached=all_segments_cached,
        )
    except EpisodeProcessingError as error:
        logger.info("节目音频转写失败：%s", error)
        return FailedEpisodeTranscriptionResult(
            episode=context.work_item,
            error=str(error),
        )


def _build_successful_episode_result(
    context: _EpisodeTranscriptionContext,
    *,
    cached: bool,
) -> SuccessfulEpisodeTranscriptionResult:
    return SuccessfulEpisodeTranscriptionResult(
        episode=replace(context.work_item, eid=context.eid),
        cached=cached,
        episode_output_dir=str(context.episode_dir),
        transcription_path=str(context.transcription_path),
        article_path=str(context.article_path),
        transcription_quality_report_path=str(context.quality_report_path),
        segment_directory=str(context.segment_dir),
    )


def _build_segment_checkpoint_store(
    context: _EpisodeTranscriptionContext,
) -> SegmentTranscriptionCheckpointStore:
    metadata = {
        "eid": context.eid,
        "podcast_title": context.work_item.podcast_title,
        "episode_url": context.work_item.episode_url,
        "title": context.work_item.title,
        "pub_date": context.work_item.pub_date,
    }
    return SegmentTranscriptionCheckpointStore(
        metadata=metadata,
        expected_metadata={
            "eid": context.eid,
            "episode_url": context.work_item.episode_url,
        },
    )


def _build_episode_context(
    *,
    work_item: EpisodeWorkItem,
    audio_output_dir: Path,
) -> _EpisodeTranscriptionContext:
    eid = work_item.resolve_episode_id()
    if eid is None:
        raise EpisodeProcessingError(
            MISSING_EPISODE_ID_ERROR.format(url=work_item.episode_url)
        )
    episode_output_dir = prepare_episode_output_directory(
        audio_output_dir,
        eid,
        podcast_title=work_item.podcast_title,
        episode_title=work_item.title,
    )
    return _EpisodeTranscriptionContext(
        work_item=work_item,
        eid=eid,
        episode_dir=episode_output_dir,
    )


def _save_episode_transcription(
    prepared_episode: DownloadedEpisodeAudio,
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
            "podcast_title": context.work_item.podcast_title,
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
