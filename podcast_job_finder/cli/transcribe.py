from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Final, NoReturn, Sequence

from podcast_job_finder.llm import (
    OpenAiCompatibleConfigError,
    load_audio_transcription_llm_runtime_config_from_env,
    load_transcription_formatting_llm_runtime_config_from_env,
)
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    VadConfig,
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.transcription_checkpoint import (
    TRANSCRIPTION_CACHE_VERSION,
    SegmentTranscriptionCheckpointStore,
    build_audio_transcription_runtime_signature,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
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


PROGRAM_NAME: Final = "podcast-transcribe"
DEFAULT_OUTPUT_DIR: Final = Path("output/transcription_segments")
INVALID_MAX_SEGMENTS_ERROR: Final = "max_segments 必须大于 0。"
LOCAL_AUDIO_SOURCE_TYPE: Final = "local_audio"


class CliUsageError(ValueError):
    """命令行参数无效时抛出的错误。"""


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    try:
        transcription_runtime = load_audio_transcription_llm_runtime_config_from_env()
        formatting_runtime = load_transcription_formatting_llm_runtime_config_from_env()
        vad_config = VadConfig()
        exported_segments = detect_and_export_speech_segments(
            args.audio_path,
            output_dir=args.output_dir,
            config=vad_config,
            overwrite=True,
        )
        selected_segments = (
            exported_segments[: args.max_segments]
            if args.max_segments is not None
            else exported_segments
        )
        source_metadata = _build_local_audio_source_metadata(args.audio_path)
        checkpoint_store = SegmentTranscriptionCheckpointStore(
            runtime_signature=build_audio_transcription_runtime_signature(
                llm_config=transcription_runtime.client_config,
                vad_config=vad_config,
            ),
            metadata=source_metadata,
            expected_metadata=source_metadata,
        )
        result, _ = transcribe_speech_segments_with_checkpoints(
            selected_segments,
            llm_client=transcription_runtime.build_client(),
            checkpoint_store=checkpoint_store,
            retry_config=transcription_runtime.retry_config,
            overwrite=args.overwrite,
        )
        article_path = args.output_dir / TRANSCRIPTION_ARTICLE_FILE_NAME
        formatted_article_path = (
            args.output_dir / FORMATTED_TRANSCRIPTION_ARTICLE_FILE_NAME
        )
        save_transcription_article(
            article_path,
            title=args.audio_path.stem,
            body=result.text,
        )
        formatted_article = format_transcription_segments(
            result.segments,
            llm_client=formatting_runtime.build_client(),
            retry_config=formatting_runtime.retry_config,
        )
        save_transcription_article(
            formatted_article_path,
            title=args.audio_path.stem,
            body=formatted_article.text,
        )
        payload = save_audio_transcription_manifest(
            args.output_dir / TRANSCRIPTION_FILE_NAME,
            metadata={
                "cache_version": TRANSCRIPTION_CACHE_VERSION,
                "runtime_signature": checkpoint_store.runtime_signature,
                **source_metadata,
                "audio_path": str(args.audio_path),
                "model": transcription_runtime.client_config.model,
                "base_url": transcription_runtime.client_config.base_url,
                "api_style": transcription_runtime.client_config.api_style,
                "formatting_model": formatting_runtime.client_config.model,
                "formatting_base_url": formatting_runtime.client_config.base_url,
                "formatting_api_style": formatting_runtime.client_config.api_style,
                "available_segment_count": len(exported_segments),
                "transcribed_segment_count": len(selected_segments),
                "article_path": str(article_path),
                "formatted_article_path": str(formatted_article_path),
                "formatting": formatted_article.to_machine_audit_dict(),
            },
            exported_segments=selected_segments,
            result=result,
        )
    except (
        AudioFileDecodeError,
        AudioSegmentExportError,
        AudioTranscriptionError,
        OpenAiCompatibleConfigError,
        OSError,
        ValueError,
        *EXPECTED_TRANSCRIPTION_FORMATTING_ERRORS,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--max-segments", type=_parse_positive_integer)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _build_local_audio_source_metadata(audio_path: Path) -> dict[str, object]:
    with audio_path.open("rb") as file_obj:
        content_sha256 = hashlib.file_digest(file_obj, "sha256").hexdigest()
    return {
        "source_type": LOCAL_AUDIO_SOURCE_TYPE,
        "source_audio_path": str(audio_path.resolve()),
        "source_audio_sha256": content_sha256,
    }


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
