from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.audio import (
    AudioFileDecodeError,
    AudioSegmentExportError,
    detect_and_export_speech_segments,
)
from podcast_job_finder.logging import configure_logging


PROGRAM_NAME: Final = "podcast-split-audio"
DEFAULT_OUTPUT_DIR: Final = Path("output/audio_segments")


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    args = _build_argument_parser().parse_args(argv)
    try:
        segments = detect_and_export_speech_segments(
            args.audio_path,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except (AudioFileDecodeError, AudioSegmentExportError, ValueError) as error:
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
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
