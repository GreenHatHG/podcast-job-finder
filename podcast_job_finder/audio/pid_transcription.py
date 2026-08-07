"""Batch podcast audio transcription with resumable segment checkpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.companies.episode_runner import EpisodeWorkItem
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    ExportedSpeechSegment,
    VadConfig,
)
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.pid_transcription_schedule import (
    DEFAULT_AUDIO_PROCESSING_MODE,
    AudioProcessingMode,
    run_audio_processing_schedule,
)
from podcast_job_finder.audio.speech_segment_checkpoint import (
    SPEECH_SEGMENT_CHECKPOINT_FILE_NAME,
    restore_or_export_speech_segments,
)
from podcast_job_finder.audio.segment_export import (
    SegmentAudioFormat,
    SpeechSegmentExportConfig,
)
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
    AudioTranscriptionResult,
)
from podcast_job_finder.audio.transcription_runtime import AudioTranscriptionRuntime
from podcast_job_finder.audio.transcription_article import (
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    save_transcription_article,
)
from podcast_job_finder.audio.transcription_manifest import (
    TRANSCRIPTION_FILE_NAME,
    TranscriptionManifestError,
    load_episode_transcription_manifest,
    save_audio_transcription_manifest,
)
from podcast_job_finder.audio.transcription_confidence_report import (
    TRANSCRIPTION_QUALITY_REPORT_FILE_NAME,
    save_transcription_quality_report,
)
from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_json,
)
from podcast_job_finder.xiaoyuzhou.episode_audio.files import (
    prepare_episode_audio_directory,
)
from podcast_job_finder.xiaoyuzhou.episode_audio.service import (
    DEFAULT_AUDIO_OUTPUT_DIR,
    EpisodeAudioDownloadError,
    EpisodeAudioDownloadResult,
    download_episode_audio,
)
from podcast_job_finder.audio.transcription_checkpoint import (
    TRANSCRIPTION_CACHE_VERSION,
    SegmentTranscriptionCheckpointStore,
    build_audio_transcription_runtime_signature,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.timestamps import build_utc_timestamp


SEGMENT_DIR_NAME: Final = "segments"
TRANSCRIPTION_REPORT_TEMPLATE: Final = "transcription_result_{pid}_{timestamp}.json"
RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"
MISSING_EPISODE_ID_ERROR: Final = "音频转写任务缺少有效的节目 ID：{url}"
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


class PidAudioTranscriptionError(RuntimeError):
    """PID 音频转写流程无法启动或保存批次结果。"""


@dataclass(slots=True, frozen=True)
class PidAudioTranscriptionRuntime:
    transcription: AudioTranscriptionRuntime
    vad_config: VadConfig = VadConfig()
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS
    strict_checkpoint_validation: bool = False

    @property
    def segment_audio_format(self) -> SegmentAudioFormat:
        return self.transcription.segment_audio_format

    @property
    def runtime_signature(self) -> str:
        return build_audio_transcription_runtime_signature(
            transcriber_signature=self.transcription.signature_payload,
            segment_audio_format=self.segment_audio_format,
            vad_config=self.vad_config,
            silence_padding_ms=self.silence_padding_ms,
        )

    def close(self) -> None:
        self.transcription.close()


@dataclass(slots=True, frozen=True)
class PidAudioTranscriptionResult:
    episode_results: list[dict[str, object]]
    success_count: int
    fail_count: int


@dataclass(slots=True, frozen=True)
class _EpisodeTranscriptionContext:
    pid: str
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
    download_result: EpisodeAudioDownloadResult


def _build_segment_checkpoint_store(
    context: _EpisodeTranscriptionContext,
    *,
    runtime: PidAudioTranscriptionRuntime,
) -> SegmentTranscriptionCheckpointStore:
    return SegmentTranscriptionCheckpointStore(
        runtime_signature=runtime.runtime_signature,
        metadata={
            "pid": context.pid,
            "eid": context.eid,
            "episode_url": context.work_item.episode_url,
            "title": context.work_item.title,
            "pub_date": context.work_item.pub_date,
        },
        expected_metadata={
            "eid": context.eid,
            "episode_url": context.work_item.episode_url,
        },
        strict_validation=runtime.strict_checkpoint_validation,
    )


def run_pid_audio_transcription(
    *,
    pid: str,
    work_items: Sequence[EpisodeWorkItem],
    runtime: PidAudioTranscriptionRuntime,
    audio_output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
    processing_mode: AudioProcessingMode = DEFAULT_AUDIO_PROCESSING_MODE,
) -> PidAudioTranscriptionResult:
    def prepare_episode(
        work_item: EpisodeWorkItem,
    ) -> _DownloadedEpisodeAudio | dict[str, object]:
        return _prepare_episode_audio(
            pid=pid,
            work_item=work_item,
            runtime=runtime,
            audio_output_dir=audio_output_dir,
        )

    def process_episode(
        prepared_episode: _DownloadedEpisodeAudio | dict[str, object],
    ) -> dict[str, object]:
        return _transcribe_prepared_episode(prepared_episode, runtime=runtime)

    episode_results = run_audio_processing_schedule(
        work_items=work_items,
        processing_mode=processing_mode,
        prepare_episode=prepare_episode,
        process_episode=process_episode,
    )
    success_count = sum(
        1 for result in episode_results if result.get("status") == RESULT_STATUS_SUCCESS
    )
    return PidAudioTranscriptionResult(
        episode_results=episode_results,
        success_count=success_count,
        fail_count=len(episode_results) - success_count,
    )


def save_pid_audio_transcription_report(
    *,
    pid: str,
    runtime: PidAudioTranscriptionRuntime,
    result: PidAudioTranscriptionResult,
    output_dir: Path,
) -> Path:
    timestamp = build_utc_timestamp()
    report_path = output_dir / TRANSCRIPTION_REPORT_TEMPLATE.format(
        pid=pid,
        timestamp=timestamp.file_label,
    )
    report = {
        "pid": pid,
        "source": "audio",
        **runtime.transcription.metadata,
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
        raise PidAudioTranscriptionError(
            SAVE_REPORT_ERROR_TEMPLATE.format(
                path=report_path,
                error_message=str(error),
            )
        ) from error
    return report_path


def _prepare_episode_audio(
    *,
    pid: str,
    work_item: EpisodeWorkItem,
    runtime: PidAudioTranscriptionRuntime,
    audio_output_dir: Path,
) -> _DownloadedEpisodeAudio | dict[str, object]:
    try:
        context = _build_episode_context(
            pid=pid,
            work_item=work_item,
            audio_output_dir=audio_output_dir,
        )
        if _can_restore_completed_transcription(context, runtime=runtime):
            logger.info("命中完整音频转写清单：eid=%s", context.eid)
            return _build_success_record(context, cached=True)

        logger.info("下载节目音频：eid=%s title=%s", context.eid, work_item.title)
        download_result = download_episode_audio(
            work_item.episode_url,
            output_dir=audio_output_dir,
        )
        return _DownloadedEpisodeAudio(
            context=context,
            download_result=download_result,
        )
    except EXPECTED_EPISODE_ERRORS as error:
        logger.info("节目音频准备失败：%s", error)
        return _build_error_record(work_item, str(error))


def _build_episode_context(
    *,
    pid: str,
    work_item: EpisodeWorkItem,
    audio_output_dir: Path,
) -> _EpisodeTranscriptionContext:
    eid = work_item.resolve_episode_id()
    if eid is None:
        raise ValueError(MISSING_EPISODE_ID_ERROR.format(url=work_item.episode_url))
    episode_output_dir = prepare_episode_audio_directory(audio_output_dir, eid)
    return _EpisodeTranscriptionContext(
        pid=pid,
        work_item=work_item,
        eid=eid,
        transcription_path=episode_output_dir / TRANSCRIPTION_FILE_NAME,
    )


def _transcribe_prepared_episode(
    prepared_episode: _DownloadedEpisodeAudio | dict[str, object],
    *,
    runtime: PidAudioTranscriptionRuntime,
) -> dict[str, object]:
    if isinstance(prepared_episode, dict):
        return prepared_episode

    context = prepared_episode.context
    try:
        logger.info(
            "转写节目音频：eid=%s title=%s", context.eid, context.work_item.title
        )
        checkpoint_store = _build_segment_checkpoint_store(
            context,
            runtime=runtime,
        )
        exported_segments = restore_or_export_speech_segments(
            source_path=prepared_episode.download_result.local_path,
            output_dir=context.segment_dir,
            checkpoint_path=context.segment_checkpoint_path,
            vad_config=runtime.vad_config,
            export_config=SpeechSegmentExportConfig(
                silence_padding_ms=runtime.silence_padding_ms,
                audio_format=runtime.segment_audio_format,
                overwrite=True,
            ),
            transcription_checkpoint_store=checkpoint_store,
        )
        transcription_result, all_segments_cached = (
            _transcribe_segments_with_checkpoints(
                exported_segments=exported_segments,
                runtime=runtime,
                checkpoint_store=checkpoint_store,
            )
        )
        _save_episode_transcription(
            context,
            runtime=runtime,
            download_result=prepared_episode.download_result,
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


def _transcribe_segments_with_checkpoints(
    exported_segments: Sequence[ExportedSpeechSegment],
    *,
    runtime: PidAudioTranscriptionRuntime,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
) -> tuple[AudioTranscriptionResult, bool]:
    result, all_segments_cached = transcribe_speech_segments_with_checkpoints(
        exported_segments,
        transcriber=runtime.transcription.transcriber,
        checkpoint_store=checkpoint_store,
    )
    return result, all_segments_cached


def _can_restore_completed_transcription(
    context: _EpisodeTranscriptionContext,
    *,
    runtime: PidAudioTranscriptionRuntime,
) -> bool:
    if not all(
        path.is_file()
        for path in (
            context.transcription_path,
            context.article_path,
            context.quality_report_path,
        )
    ):
        return False
    try:
        manifest = load_episode_transcription_manifest(context.transcription_path)
    except TranscriptionManifestError as error:
        logger.warning("读取完整音频转写清单失败，将继续处理：%s", error)
        return False
    expected_metadata = {
        "cache_version": TRANSCRIPTION_CACHE_VERSION,
        "runtime_signature": runtime.runtime_signature,
        "eid": context.eid,
        "episode_url": context.work_item.episode_url,
    }
    return all(
        manifest.metadata.get(field_name) == expected_value
        for field_name, expected_value in expected_metadata.items()
    )


def _save_episode_transcription(
    context: _EpisodeTranscriptionContext,
    *,
    runtime: PidAudioTranscriptionRuntime,
    download_result: EpisodeAudioDownloadResult,
    result: AudioTranscriptionResult,
    exported_segments: Sequence[ExportedSpeechSegment],
) -> None:
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
            "cache_version": TRANSCRIPTION_CACHE_VERSION,
            "runtime_signature": runtime.runtime_signature,
            "pid": context.pid,
            "eid": context.eid,
            "title": context.work_item.title,
            "pub_date": context.work_item.pub_date,
            "episode_url": context.work_item.episode_url,
            **runtime.transcription.metadata,
            "segment_audio_format": runtime.segment_audio_format,
            "audio_path": str(download_result.local_path),
            "source_url": download_result.source_url,
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
