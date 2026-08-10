"""Batch podcast audio transcription with resumable segment checkpoints."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

from podcast_job_finder.companies.episode_runner import EpisodeWorkItem
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    ExportedSpeechSegment,
    VadConfig,
)
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.batch_transcription_schedule import (
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
    TranscribedSpeechSegment,
)
from podcast_job_finder.audio.transcription_runtime import AudioTranscriptionRuntime
from podcast_job_finder.audio.transcription_article import (
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    build_transcription_article,
    save_transcription_article,
)
from podcast_job_finder.audio.transcription_manifest import (
    TRANSCRIPTION_FILE_NAME,
    TranscriptionManifestError,
    load_episode_transcription_manifest,
    parse_transcribed_segment,
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
from podcast_job_finder.audio.transcription_checkpoint import (
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
class BatchAudioTranscriptionRuntime:
    transcription: AudioTranscriptionRuntime
    vad_config: VadConfig = VadConfig()
    silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS

    @property
    def segment_audio_format(self) -> SegmentAudioFormat:
        return self.transcription.segment_audio_format

    def close(self) -> None:
        self.transcription.close()


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
    runtime: BatchAudioTranscriptionRuntime,
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
    runtime: BatchAudioTranscriptionRuntime,
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
        if resume and _can_restore_completed_transcription(context):
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
    runtime: BatchAudioTranscriptionRuntime,
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
            export_config=SpeechSegmentExportConfig(
                silence_padding_ms=runtime.silence_padding_ms,
                audio_format=runtime.segment_audio_format,
                overwrite=True,
            ),
            resume=resume,
        )
        transcription_result, all_segments_cached = (
            _transcribe_segments_with_checkpoints(
                exported_segments=exported_segments,
                runtime=runtime,
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


def _transcribe_segments_with_checkpoints(
    exported_segments: Sequence[ExportedSpeechSegment],
    *,
    runtime: BatchAudioTranscriptionRuntime,
    checkpoint_store: SegmentTranscriptionCheckpointStore,
    resume: bool,
) -> tuple[AudioTranscriptionResult, bool]:
    result, all_segments_cached = transcribe_speech_segments_with_checkpoints(
        exported_segments,
        transcriber=runtime.transcription.transcriber,
        checkpoint_store=checkpoint_store,
        resume=resume,
    )
    return result, all_segments_cached


def _can_restore_completed_transcription(
    context: _EpisodeTranscriptionContext,
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
        _validate_completed_transcription_artifacts(context, manifest.metadata)
    except (
        OSError,
        TranscriptionManifestError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        logger.warning("读取完整音频转写清单失败，将继续处理：%s", error)
        return False
    expected_metadata = {
        "eid": context.eid,
        "episode_url": context.work_item.episode_url,
    }
    return all(
        manifest.metadata.get(field_name) == expected_value
        for field_name, expected_value in expected_metadata.items()
    )


def _validate_completed_transcription_artifacts(
    context: _EpisodeTranscriptionContext,
    metadata: dict[str, object] | Mapping[str, object],
) -> None:
    raw_segments = metadata.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("完整音频转写清单缺少 segments 数组。")
    if metadata.get("segment_count") != len(raw_segments):
        raise ValueError("完整音频转写清单中的 segment_count 已变化。")
    text = metadata.get("text")
    if not isinstance(text, str):
        raise ValueError("完整音频转写清单中的 text 必须是字符串。")
    audio_path_value = metadata.get("audio_path")
    if not isinstance(audio_path_value, str) or not _is_non_empty_file(
        Path(audio_path_value)
    ):
        raise ValueError("完整音频转写清单中的源音频文件无效。")
    _validate_quality_report(context.quality_report_path, len(raw_segments))
    article_text = context.article_path.read_text(encoding="utf-8")
    expected_article = build_transcription_article(
        title=context.work_item.title or context.eid,
        body=text,
    )
    if article_text != expected_article:
        raise ValueError("完整音频转写文章与清单内容不一致。")
    parsed_segments = _validate_manifest_segments(context, raw_segments)
    if text != "\n".join(segment.text for segment in parsed_segments):
        raise ValueError("完整音频转写清单中的 text 与 segments 不一致。")


def _validate_manifest_segments(
    context: _EpisodeTranscriptionContext,
    raw_segments: list[object],
) -> list[TranscribedSpeechSegment]:
    parsed_segments: list[TranscribedSpeechSegment] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ValueError(f"完整音频转写清单中的片段无效：index={index}")
        parsed_segment = parse_transcribed_segment(
            raw_segment,
            path=context.transcription_path,
            index=index,
        )
        if parsed_segment.index != index + 1:
            raise ValueError(
                "完整音频转写片段编号不连续："
                f"expected_index={index + 1} actual_index={parsed_segment.index}"
            )
        parsed_segments.append(parsed_segment)
        _validate_manifest_segment_artifacts(
            raw_segment,
            expected_segment=parsed_segment,
            segment_dir=context.segment_dir,
        )
    return parsed_segments


def _validate_manifest_segment_artifacts(
    raw_segment: dict[str, object],
    *,
    expected_segment: TranscribedSpeechSegment,
    segment_dir: Path,
) -> None:
    audio_path = _require_artifact_path(raw_segment, "audio_path")
    transcription_path = _require_artifact_path(raw_segment, "transcription_path")
    if audio_path.resolve().parent != segment_dir.resolve():
        raise ValueError(
            f"完整音频转写片段不在预期目录中：index={expected_segment.index}"
        )
    if transcription_path.resolve() != audio_path.with_suffix(".json").resolve():
        raise ValueError(f"完整音频转写片段路径不对应：index={expected_segment.index}")
    if not _is_non_empty_file(audio_path) or not _is_non_empty_file(transcription_path):
        raise ValueError(f"完整音频转写片段文件无效：index={expected_segment.index}")
    segment_payload = _read_json_object(transcription_path)
    parsed_checkpoint = parse_transcribed_segment(
        segment_payload,
        path=transcription_path,
        index=expected_segment.index,
    )
    if parsed_checkpoint != expected_segment:
        raise ValueError(f"完整音频转写片段内容不一致：index={expected_segment.index}")


def _validate_quality_report(path: Path, expected_segment_count: int) -> None:
    payload = _read_json_object(path)
    if payload.get("segment_count") != expected_segment_count:
        raise ValueError("音频转写质量报告中的 segment_count 已变化。")


def _read_json_object(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件必须是对象：{path}")
    return payload


def _require_artifact_path(payload: dict[str, object], field_name: str) -> Path:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"完整音频转写片段缺少 {field_name}。")
    return Path(value)


def _is_non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _save_episode_transcription(  # pylint: disable=too-many-arguments
    prepared_episode: _DownloadedEpisodeAudio,
    *,
    runtime: BatchAudioTranscriptionRuntime,
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
            **runtime.transcription.metadata,
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
