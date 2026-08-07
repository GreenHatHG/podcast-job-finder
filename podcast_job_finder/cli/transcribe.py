from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Final, NoReturn, Sequence

from podcast_job_finder.llm import (
    LlmRuntimeConfig,
    OpenAiCompatibleConfigError,
    load_transcription_formatting_llm_runtime_config_from_env,
)
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.transcription_checkpoint import (
    TRANSCRIPTION_CACHE_VERSION,
    SegmentTranscriptionCheckpointStore,
    build_audio_transcription_runtime_signature,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.audio.segment_export import (
    MP3_SEGMENT_AUDIO_FORMAT,
    WAV_SEGMENT_AUDIO_FORMAT,
    SegmentAudioFormat,
    SpeechSegmentExportConfig,
    parse_segment_audio_format,
)
from podcast_job_finder.audio.speech_pipeline import DEFAULT_SILENCE_PADDING_MS
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
    TranscribedSpeechSegment,
)
from podcast_job_finder.audio.transcription_runtime import (
    AudioTranscriptionConfigError,
    AudioTranscriptionRuntime,
    load_audio_transcription_runtime_from_env,
)
from podcast_job_finder.audio.transcription_manifest import (
    TRANSCRIPTION_FILE_NAME,
    save_audio_transcription_manifest,
)
from podcast_job_finder.audio.transcription_article import (
    FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME,
    TRANSCRIPTION_ARTICLE_FILE_NAME,
    save_transcription_article,
)
from podcast_job_finder.audio.transcription_formatter import (
    EXPECTED_TRANSCRIPTION_FORMATTING_ERRORS,
    format_transcription_segments,
)
from podcast_job_finder.audio.transcription_confidence_report import (
    TRANSCRIPTION_QUALITY_REPORT_FILE_NAME,
    save_transcription_quality_report,
)


PROGRAM_NAME: Final = "podcast-transcribe"
DEFAULT_OUTPUT_DIR: Final = Path("output/transcription_segments")
INVALID_MAX_SEGMENTS_ERROR: Final = "max_segments 必须大于 0。"
LOCAL_AUDIO_SOURCE_TYPE: Final = "local_audio"
AUTO_SEGMENT_AUDIO_FORMAT: Final = "auto"
SEGMENT_AUDIO_FORMAT_CHOICES: Final = (
    AUTO_SEGMENT_AUDIO_FORMAT,
    WAV_SEGMENT_AUDIO_FORMAT,
    MP3_SEGMENT_AUDIO_FORMAT,
)


class CliUsageError(ValueError):
    """命令行参数无效时抛出的错误。"""


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    transcription_runtime = None
    try:
        transcription_runtime = load_audio_transcription_runtime_from_env()
        payload = _run_transcription(args, transcription_runtime)
    except (
        AudioFileDecodeError,
        AudioSegmentExportError,
        AudioTranscriptionConfigError,
        AudioTranscriptionError,
        OpenAiCompatibleConfigError,
        OSError,
        ValueError,
        *EXPECTED_TRANSCRIPTION_FORMATTING_ERRORS,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if transcription_runtime is not None:
            transcription_runtime.close()

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_transcription(
    args: argparse.Namespace,
    transcription_runtime: AudioTranscriptionRuntime,
) -> dict[str, object]:
    formatting_runtime = load_transcription_formatting_llm_runtime_config_from_env()
    segment_audio_format = _resolve_segment_audio_format(
        args.segment_audio_format,
        transcription_runtime=transcription_runtime,
    )
    vad_config = transcription_runtime.vad_config
    exported_segments = detect_and_export_speech_segments(
        args.audio_path,
        output_dir=args.output_dir,
        config=vad_config,
        export_config=SpeechSegmentExportConfig(
            silence_padding_ms=DEFAULT_SILENCE_PADDING_MS,
            audio_format=segment_audio_format,
            overwrite=True,
        ),
    )
    selected_segments = (
        exported_segments[: args.max_segments]
        if args.max_segments is not None
        else exported_segments
    )
    source_metadata = _build_local_audio_source_metadata(args.audio_path)
    checkpoint_store = SegmentTranscriptionCheckpointStore(
        runtime_signature=build_audio_transcription_runtime_signature(
            transcriber_signature=transcription_runtime.signature_payload,
            segment_audio_format=segment_audio_format,
            vad_config=vad_config,
        ),
        metadata=source_metadata,
        expected_metadata=source_metadata,
        strict_validation=args.strict_checkpoint_validation,
    )
    result, _ = transcribe_speech_segments_with_checkpoints(
        selected_segments,
        transcriber=transcription_runtime.transcriber,
        checkpoint_store=checkpoint_store,
        overwrite=args.overwrite,
    )
    article_path = args.output_dir / TRANSCRIPTION_ARTICLE_FILE_NAME
    save_transcription_article(
        article_path,
        title=args.audio_path.stem,
        body=result.text,
    )
    quality_report_path = args.output_dir / TRANSCRIPTION_QUALITY_REPORT_FILE_NAME
    save_transcription_quality_report(
        quality_report_path,
        result,
        exported_segments=selected_segments,
    )
    formatting_metadata = _format_transcription(
        args,
        result.segments,
        formatting_runtime=formatting_runtime,
    )
    return save_audio_transcription_manifest(
        args.output_dir / TRANSCRIPTION_FILE_NAME,
        metadata={
            "cache_version": TRANSCRIPTION_CACHE_VERSION,
            "runtime_signature": checkpoint_store.runtime_signature,
            **source_metadata,
            **transcription_runtime.metadata,
            "audio_path": str(args.audio_path),
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
    args: argparse.Namespace,
    segments: Sequence[TranscribedSpeechSegment],
    *,
    formatting_runtime: LlmRuntimeConfig,
) -> dict[str, object]:
    formatted_article = format_transcription_segments(
        segments,
        llm_client=formatting_runtime.build_client(),
        retry_config=formatting_runtime.retry_config,
    )
    formatted_article_path = args.output_dir / FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME
    save_transcription_article(
        formatted_article_path,
        title=args.audio_path.stem,
        body=formatted_article.text,
    )
    return {
        "formatting_model": formatting_runtime.client_config.model,
        "formatting_base_url": formatting_runtime.client_config.base_url,
        "formatting_api_style": formatting_runtime.client_config.api_style,
        "formatted_article_path": str(formatted_article_path),
        "formatting": formatted_article.to_machine_audit_dict(),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--max-segments", type=_parse_positive_integer)
    parser.add_argument(
        "--segment-audio-format",
        choices=SEGMENT_AUDIO_FORMAT_CHOICES,
        default=AUTO_SEGMENT_AUDIO_FORMAT,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict-checkpoint-validation",
        action="store_true",
    )
    return parser


def _build_local_audio_source_metadata(audio_path: Path) -> dict[str, object]:
    with audio_path.open("rb") as file_obj:
        content_sha256 = hashlib.file_digest(file_obj, "sha256").hexdigest()
    return {
        "source_type": LOCAL_AUDIO_SOURCE_TYPE,
        "source_audio_path": str(audio_path.resolve()),
        "source_audio_sha256": content_sha256,
    }


def _resolve_segment_audio_format(
    value: str,
    *,
    transcription_runtime: AudioTranscriptionRuntime,
) -> SegmentAudioFormat:
    if value == AUTO_SEGMENT_AUDIO_FORMAT:
        return transcription_runtime.segment_audio_format
    return parse_segment_audio_format(value)


def _parse_positive_integer(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError:
        _raise_argument_type_error()
    if value <= 0:
        _raise_argument_type_error()
    return value


def _raise_argument_type_error() -> NoReturn:
    raise argparse.ArgumentTypeError(INVALID_MAX_SEGMENTS_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
