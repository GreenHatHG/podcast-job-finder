"""本地音频的切分、转写、格式化和结果保存流程。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from podcast_job_finder.audio.segmentation.segment_export import (
    SegmentAudioFormat,
)
from podcast_job_finder.audio.segmentation.speech_segment_checkpoint import (
    SPEECH_SEGMENT_CHECKPOINT_FILE_NAME,
    restore_or_export_speech_segments,
)
from podcast_job_finder.llm import LlmRuntime
from podcast_job_finder.transcription.checkpoint import (
    SegmentTranscriptionCheckpointStore,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.transcription.formatting.article import (
    FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME,
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    save_transcription_article,
)
from podcast_job_finder.transcription.formatting.formatter import (
    format_transcription_segments,
)
from podcast_job_finder.transcription.manifest import (
    TRANSCRIPTION_FILE_NAME,
    save_audio_transcription_manifest,
)
from podcast_job_finder.transcription.models import (
    AudioTranscriptionRuntime,
    TranscribedSpeechSegment,
)
from podcast_job_finder.transcription.quality_report import (
    TRANSCRIPTION_QUALITY_REPORT_FILE_NAME,
    save_transcription_quality_report,
)


LOCAL_AUDIO_SOURCE_TYPE = "local_audio"


def transcribe_local_audio(  # pylint: disable=too-many-arguments
    audio_path: Path,
    *,
    output_dir: Path,
    max_segments: int | None,
    segment_audio_format: SegmentAudioFormat,
    resume: bool,
    transcription_runtime: AudioTranscriptionRuntime,
    formatting_runtime: LlmRuntime,
) -> dict[str, object]:
    """处理本地音频并返回写入转写清单的完整内容。"""

    source_metadata = _build_source_metadata(audio_path)
    checkpoint_store = SegmentTranscriptionCheckpointStore(
        metadata=source_metadata,
        expected_metadata=source_metadata,
    )
    exported_segments = restore_or_export_speech_segments(
        source_path=audio_path,
        output_dir=output_dir,
        checkpoint_path=output_dir / SPEECH_SEGMENT_CHECKPOINT_FILE_NAME,
        vad_config=transcription_runtime.vad_config,
        silence_padding_ms=transcription_runtime.silence_padding_ms,
        audio_format=segment_audio_format,
        overwrite=True,
        resume=resume,
    )
    selected_segments = (
        exported_segments[:max_segments]
        if max_segments is not None
        else exported_segments
    )
    result, _ = transcribe_speech_segments_with_checkpoints(
        selected_segments,
        transcriber=transcription_runtime.transcriber,
        checkpoint_store=checkpoint_store,
        resume=resume,
    )
    article_path = output_dir / TRANSCRIPTION_ARTICLE_FILE_NAME
    save_transcription_article(
        article_path,
        title=audio_path.stem,
        body=result.text,
    )
    quality_report_path = output_dir / TRANSCRIPTION_QUALITY_REPORT_FILE_NAME
    save_transcription_quality_report(
        quality_report_path,
        result,
        exported_segments=selected_segments,
    )
    formatting_metadata = _format_transcription(
        result.segments,
        output_dir=output_dir,
        article_title=audio_path.stem,
        formatting_runtime=formatting_runtime,
    )
    return save_audio_transcription_manifest(
        output_dir / TRANSCRIPTION_FILE_NAME,
        metadata={
            **source_metadata,
            **transcription_runtime.metadata,
            "audio_path": str(audio_path),
            "segment_audio_format": segment_audio_format,
            "available_segment_count": len(exported_segments),
            "transcribed_segment_count": len(selected_segments),
            "article_path": str(article_path),
            "transcription_quality_report_path": str(quality_report_path),
            **formatting_metadata,
        },
        exported_segments=selected_segments,
        result=result,
    )


def _format_transcription(
    segments: Sequence[TranscribedSpeechSegment],
    *,
    output_dir: Path,
    article_title: str,
    formatting_runtime: LlmRuntime,
) -> dict[str, object]:
    formatted_article = format_transcription_segments(
        segments,
        llm_client=formatting_runtime.client,
        retry_config=formatting_runtime.retry_config,
        max_workers=formatting_runtime.max_in_flight_requests,
    )
    formatted_article_path = output_dir / FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME
    save_transcription_article(
        formatted_article_path,
        title=article_title,
        body=formatted_article.text,
    )
    return {
        "formatting_model": formatting_runtime.model,
        "formatting_base_url": formatting_runtime.base_url,
        "formatting_api_style": formatting_runtime.api_style,
        "formatted_article_path": str(formatted_article_path),
        "formatting": formatted_article.to_machine_audit_dict(),
    }


def _build_source_metadata(audio_path: Path) -> dict[str, object]:
    return {
        "source_type": LOCAL_AUDIO_SOURCE_TYPE,
        "source_audio_path": str(audio_path.resolve()),
    }
