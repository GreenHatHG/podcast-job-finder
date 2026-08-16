from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.audio import detect_and_export_speech_segments
from podcast_job_finder.errors import PodcastJobFinderError
from podcast_job_finder.logging import configure_logging
from podcast_job_finder.audio.segmentation.segment_export import (
    DEFAULT_SEGMENT_AUDIO_FORMAT,
    SUPPORTED_SEGMENT_AUDIO_FORMATS,
    parse_segment_audio_format,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.output_paths import LOCAL_AUDIO_SEGMENTS_OUTPUT_DIR


PROGRAM_NAME: Final = "podcast-split-audio"


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    try:
        segments = detect_and_export_speech_segments(
            args.audio_path,
            output_dir=args.output_dir,
            silence_padding_ms=DEFAULT_SILENCE_PADDING_MS,
            audio_format=parse_segment_audio_format(args.segment_audio_format),
            overwrite=args.overwrite,
        )
    except PodcastJobFinderError as error:
        print(str(error), file=sys.stderr)
        return 1

    payload = {
        "audio_path": str(args.audio_path),
        "output_dir": str(args.output_dir),
        "segment_count": len(segments),
        "segments": [segment.to_dict() for segment in segments],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LOCAL_AUDIO_SEGMENTS_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--segment-audio-format",
        choices=sorted(SUPPORTED_SEGMENT_AUDIO_FORMATS),
        default=DEFAULT_SEGMENT_AUDIO_FORMAT,
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
