"""Batch podcast audio transcription with resumable segment checkpoints."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.companies.episode_runner import EpisodeWorkItem
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    LlmRetryConfig,
    OpenAiCompatibleConfig,
    OpenAiCompatibleLlmError,
)
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    ExportedSpeechSegment,
    VadConfig,
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.transcription import (
    TRANSCRIPTION_PROMPT_TEMPLATE,
    AudioTranscriptionClientProtocol,
    AudioTranscriptionError,
    AudioTranscriptionResult,
    TranscribedSpeechSegment,
    transcribe_speech_segment,
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
from podcast_job_finder.runtime_signature import build_runtime_signature_hash
from podcast_job_finder.audio.transcription_checkpoint import (
    SegmentTranscriptionCheckpointStore,
)
from podcast_job_finder.timestamps import build_utc_timestamp


TRANSCRIPTION_FILE_NAME: Final = "transcription.json"
SEGMENT_DIR_NAME: Final = "segments"
TRANSCRIPTION_REPORT_TEMPLATE: Final = "transcription_result_{pid}_{timestamp}.json"
TRANSCRIPTION_CACHE_VERSION: Final = 3
RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"
MISSING_EPISODE_ID_ERROR: Final = "音频转写任务缺少有效的节目 ID：{url}"
SAVE_REPORT_ERROR_TEMPLATE: Final = "保存音频转写批次报告失败：{path}，{error_message}"
SAVE_TRANSCRIPTION_ERROR_TEMPLATE: Final = "保存节目转写失败：{path}，{error_message}"

logger = logging.getLogger(__name__)

EXPECTED_EPISODE_ERRORS = (
    AudioFileDecodeError,
    AudioSegmentExportError,
    AudioTranscriptionError,
    EmptyLlmResponseError,
    EpisodeAudioDownloadError,
    OpenAiCompatibleLlmError,
    OSError,
    ValueError,
)


class PidAudioTranscriptionError(RuntimeError):
    """PID 音频转写流程无法启动或保存批次结果。"""


@dataclass(slots=True, frozen=True)
class PidAudioTranscriptionRuntime:
    llm_client: AudioTranscriptionClientProtocol
    retry_config: LlmRetryConfig
    llm_config: OpenAiCompatibleConfig
    vad_config: VadConfig = VadConfig()
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS

    @property
    def runtime_signature(self) -> str:
        return build_audio_transcription_runtime_signature(self)


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


def build_audio_transcription_runtime_signature(
    runtime: PidAudioTranscriptionRuntime,
) -> str:
    signature_payload = {
        "cache_version": TRANSCRIPTION_CACHE_VERSION,
        "model": runtime.llm_config.model,
        "base_url": runtime.llm_config.base_url,
        "api_style": runtime.llm_config.api_style,
        "prompt_template": TRANSCRIPTION_PROMPT_TEMPLATE,
        "vad_config": asdict(runtime.vad_config),
        "silence_padding_ms": runtime.silence_padding_ms,
    }
    return build_runtime_signature_hash(signature_payload)


def _build_segment_checkpoint_store(
    context: _EpisodeTranscriptionContext,
    *,
    runtime_signature: str,
) -> SegmentTranscriptionCheckpointStore:
    return SegmentTranscriptionCheckpointStore(
        cache_version=TRANSCRIPTION_CACHE_VERSION,
        runtime_signature=runtime_signature,
        pid=context.pid,
        eid=context.eid,
        episode_url=context.work_item.episode_url,
        title=context.work_item.title,
        pub_date=context.work_item.pub_date,
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
        "model": runtime.llm_config.model,
        "base_url": runtime.llm_config.base_url,
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
            silence_padding_ms=runtime.silence_padding_ms,
            overwrite=True,
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
    transcribed_segments: list[TranscribedSpeechSegment] = []
    previous_text = ""
    all_segments_cached = manifest_exists and bool(exported_segments)
    checkpoint_store = _build_segment_checkpoint_store(
        context,
        runtime_signature=runtime.runtime_signature,
    )
    for exported_segment in exported_segments:
        transcription_path = exported_segment.file_path.with_suffix(".json")
        transcribed_segment = checkpoint_store.load(
            transcription_path,
            exported_segment=exported_segment,
            previous_text=previous_text,
        )
        if transcribed_segment is None:
            all_segments_cached = False
            transcribed_segment = transcribe_speech_segment(
                exported_segment,
                llm_client=runtime.llm_client,
                previous_text=previous_text,
                retry_config=runtime.retry_config,
            )
            checkpoint_store.save(
                transcription_path,
                exported_segment=exported_segment,
                transcribed_segment=transcribed_segment,
                previous_text=previous_text,
            )
        else:
            logger.info(
                "命中音频片段转写检查点：eid=%s index=%d",
                context.eid,
                exported_segment.index,
            )
        transcribed_segments.append(transcribed_segment)
        previous_text = transcribed_segment.text
    return (
        AudioTranscriptionResult(segments=transcribed_segments),
        all_segments_cached,
    )


def _save_episode_transcription(
    context: _EpisodeTranscriptionContext,
    *,
    runtime: PidAudioTranscriptionRuntime,
    download_result: EpisodeAudioDownloadResult,
    result: AudioTranscriptionResult,
    exported_segments: Sequence[ExportedSpeechSegment],
) -> None:
    created_at = build_utc_timestamp().text
    segment_records = _build_segment_records(exported_segments, result)
    payload = {
        "cache_version": TRANSCRIPTION_CACHE_VERSION,
        "runtime_signature": runtime.runtime_signature,
        "pid": context.pid,
        "eid": context.eid,
        "title": context.work_item.title,
        "pub_date": context.work_item.pub_date,
        "episode_url": context.work_item.episode_url,
        "model": runtime.llm_config.model,
        "base_url": runtime.llm_config.base_url,
        "api_style": runtime.llm_config.api_style,
        "audio_path": str(download_result.local_path),
        "source_url": download_result.source_url,
        "created_at": created_at,
        "segment_count": len(segment_records),
        "text": result.text,
        "segments": segment_records,
    }
    path = context.transcription_path
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            path,
            payload,
            mode=DEFAULT_FILE_CREATION_MODE,
        )
    except OSError as error:
        raise OSError(
            SAVE_TRANSCRIPTION_ERROR_TEMPLATE.format(
                path=path,
                error_message=str(error),
            )
        ) from error


def _build_segment_records(
    exported_segments: Sequence[ExportedSpeechSegment],
    result: AudioTranscriptionResult,
) -> list[dict[str, object]]:
    exported_by_index = {segment.index: segment for segment in exported_segments}
    records: list[dict[str, object]] = []
    for transcribed_segment in result.segments:
        exported_segment = exported_by_index.get(transcribed_segment.index)
        if exported_segment is None:
            raise ValueError(
                f"音频转写结果缺少对应的已导出片段：index={transcribed_segment.index}"
            )
        records.append(
            {
                "index": transcribed_segment.index,
                "start_ms": transcribed_segment.start_ms,
                "end_ms": transcribed_segment.end_ms,
                "audio_path": str(exported_segment.file_path),
                "transcription_path": str(
                    exported_segment.file_path.with_suffix(".json")
                ),
                "text": transcribed_segment.text,
            }
        )
    return records


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
