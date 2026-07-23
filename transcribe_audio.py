from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, NoReturn, Sequence

from openai_compatible_llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmClient,
    OpenAiCompatibleLlmError,
    load_openai_compatible_config_from_env,
)
from logging_config import configure_logging
from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
    transcribe_speech_segments,
)


PROGRAM_NAME: Final = "python transcribe_audio.py"
DEFAULT_SEGMENT_OUTPUT_DIR: Final = Path("output/transcription_segments")
INVALID_MAX_SEGMENTS_ERROR: Final = "max_segments 必须大于 0。"


class CliUsageError(ValueError):
    """命令行参数无效时抛出的错误。"""


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    try:
        config = load_openai_compatible_config_from_env()
        exported_segments = detect_and_export_speech_segments(
            args.audio_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        selected_segments = (
            exported_segments[: args.max_segments]
            if args.max_segments is not None
            else exported_segments
        )
        result = transcribe_speech_segments(
            selected_segments,
            llm_client=OpenAiCompatibleLlmClient(config),
        )
    except (
        AudioFileDecodeError,
        AudioSegmentExportError,
        AudioTranscriptionError,
        EmptyLlmResponseError,
        OpenAiCompatibleConfigError,
        OpenAiCompatibleLlmError,
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
