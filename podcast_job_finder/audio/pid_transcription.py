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
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
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
    save_audio_transcription_manifest,
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


def _build_segment_checkpoint_store(
    context: _EpisodeTranscriptionContext,
    *,
    runtime_signature: str,
) -> SegmentTranscriptionCheckpointStore:
    return SegmentTranscriptionCheckpointStore(
        runtime_signature=runtime_signature,
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
    )


def run_pid_audio_transcription(
    *,
    pid: str,
    work_items: Sequence[EpisodeWorkItem],
    runtime: PidAudioTranscriptionRuntime,
    audio_output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
) -> PidAudioTranscriptionResult:
    episode_results = [
        _run_episode_audio_transcription(
            pid=pid,
            work_item=work_item,
            runtime=runtime,
            audio_output_dir=audio_output_dir,
        )
        for work_item in work_items
    ]
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


def _run_episode_audio_transcription(
    *,
    pid: str,
    work_item: EpisodeWorkItem,
    runtime: PidAudioTranscriptionRuntime,
    audio_output_dir: Path,
) -> dict[str, object]:
    try:
        eid = work_item.resolve_episode_id()
        if eid is None:
            raise ValueError(MISSING_EPISODE_ID_ERROR.format(url=work_item.episode_url))
        episode_output_dir = prepare_episode_audio_directory(
            audio_output_dir,
            eid,
        )
        context = _EpisodeTranscriptionContext(
            pid=pid,
            work_item=work_item,
            eid=eid,
            transcription_path=episode_output_dir / TRANSCRIPTION_FILE_NAME,
        )
        had_transcription_manifest = context.transcription_path.is_file()

        logger.info("下载并转写节目音频：eid=%s title=%s", eid, work_item.title)
        download_result = download_episode_audio(
            work_item.episode_url,
            output_dir=audio_output_dir,
        )
        exported_segments = detect_and_export_speech_segments(
            download_result.local_path,
            output_dir=episode_output_dir / SEGMENT_DIR_NAME,
            config=runtime.vad_config,
            export_config=SpeechSegmentExportConfig(
                silence_padding_ms=runtime.silence_padding_ms,
                audio_format=runtime.segment_audio_format,
                overwrite=True,
            ),
        )
        transcription_result, all_segments_cached = (
            _transcribe_segments_with_checkpoints(
                context=context,
                exported_segments=exported_segments,
                runtime=runtime,
                manifest_exists=had_transcription_manifest,
            )
        )
        _save_episode_transcription(
            context,
            runtime=runtime,
            download_result=download_result,
            result=transcription_result,
            exported_segments=exported_segments,
        )
        return _build_success_record(
            context,
            cached=all_segments_cached,
        )
    except EXPECTED_EPISODE_ERRORS as error:
        logger.info("节目音频转写失败：%s", error)
        return _build_error_record(work_item, str(error))


def _transcribe_segments_with_checkpoints(
    context: _EpisodeTranscriptionContext,
    exported_segments: Sequence[ExportedSpeechSegment],
    *,
    runtime: PidAudioTranscriptionRuntime,
    manifest_exists: bool,
) -> tuple[AudioTranscriptionResult, bool]:
    checkpoint_store = _build_segment_checkpoint_store(
        context,
        runtime_signature=runtime.runtime_signature,
    )
    result, all_segments_cached = transcribe_speech_segments_with_checkpoints(
        exported_segments,
        transcriber=runtime.transcription.transcriber,
        checkpoint_store=checkpoint_store,
    )
    return result, manifest_exists and all_segments_cached


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
            "segment_directory": str(
                context.transcription_path.parent / SEGMENT_DIR_NAME
            ),
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
