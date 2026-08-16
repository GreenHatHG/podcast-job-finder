from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, NoReturn, Sequence

from podcast_job_finder.llm import load_transcription_formatting_llm_runtime_from_env
from podcast_job_finder.errors import PodcastJobFinderError
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.audio.segmentation.segment_export import (
    MP3_SEGMENT_AUDIO_FORMAT,
    WAV_SEGMENT_AUDIO_FORMAT,
    SegmentAudioFormat,
    parse_segment_audio_format,
)
from podcast_job_finder.transcription.runtime import (
    AudioTranscriptionRuntime,
    load_audio_transcription_runtime_from_env,
)
from podcast_job_finder.transcription import local_audio
from podcast_job_finder.transcription.local_audio import transcribe_local_audio


PROGRAM_NAME: Final = "podcast-transcribe"
DEFAULT_OUTPUT_DIR: Final = Path("output/transcription_segments")
INVALID_MAX_SEGMENTS_ERROR: Final = "max_segments 必须大于 0。"
LOCAL_AUDIO_SOURCE_TYPE = local_audio.LOCAL_AUDIO_SOURCE_TYPE
AUTO_SEGMENT_AUDIO_FORMAT: Final = "auto"
SEGMENT_AUDIO_FORMAT_CHOICES: Final = (
    AUTO_SEGMENT_AUDIO_FORMAT,
    WAV_SEGMENT_AUDIO_FORMAT,
    MP3_SEGMENT_AUDIO_FORMAT,
)


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    transcription_runtime = None
    try:
        transcription_runtime = load_audio_transcription_runtime_from_env()
        formatting_runtime = load_transcription_formatting_llm_runtime_from_env()
        payload = transcribe_local_audio(
            args.audio_path,
            output_dir=args.output_dir,
            max_segments=args.max_segments,
            segment_audio_format=_resolve_segment_audio_format(
                args.segment_audio_format,
                transcription_runtime=transcription_runtime,
            ),
            resume=args.resume,
            transcription_runtime=transcription_runtime,
            formatting_runtime=formatting_runtime,
        )
    except PodcastJobFinderError as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        if transcription_runtime is not None:
            transcription_runtime.close()

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
    parser.add_argument(
        "--segment-audio-format",
        choices=SEGMENT_AUDIO_FORMAT_CHOICES,
        default=AUTO_SEGMENT_AUDIO_FORMAT,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


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
