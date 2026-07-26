from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Final, NoReturn, Sequence

from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
    load_audio_transcription_llm_runtime_config_from_env,
)
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    VadConfig,
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.transcription_checkpoint import (
    SegmentTranscriptionCheckpointStore,
    build_audio_transcription_runtime_signature,
    transcribe_speech_segments_with_checkpoints,
)
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
)


PROGRAM_NAME: Final = "podcast-transcribe"
DEFAULT_SEGMENT_OUTPUT_DIR: Final = Path("output/transcription_segments")
INVALID_MAX_SEGMENTS_ERROR: Final = "max_segments 必须大于 0。"
LOCAL_AUDIO_SOURCE_TYPE: Final = "local_audio"


class CliUsageError(ValueError):
    """命令行参数无效时抛出的错误。"""


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    try:
        llm_runtime = load_audio_transcription_llm_runtime_config_from_env()
        config = llm_runtime.client_config
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
                llm_config=config,
                vad_config=vad_config,
            ),
            metadata=source_metadata,
            expected_metadata=source_metadata,
        )
        result, _ = transcribe_speech_segments_with_checkpoints(
            selected_segments,
            llm_client=llm_runtime.build_client(),
            checkpoint_store=checkpoint_store,
            retry_config=llm_runtime.retry_config,
            overwrite=args.overwrite,
        )
    except (
        AudioFileDecodeError,
        AudioSegmentExportError,
        AudioTranscriptionError,
        EmptyLlmResponseError,
        OpenAiCompatibleConfigError,
        OpenAiCompatibleLlmError,
        OSError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = {
        "audio_path": str(args.audio_path),
        "model": config.model,
        "available_segment_count": len(exported_segments),
        "transcribed_segment_count": len(selected_segments),
        **result.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SEGMENT_OUTPUT_DIR,
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
