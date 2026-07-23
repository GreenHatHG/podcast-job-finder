from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.xiaoyuzhou.episode_audio.service import (
    DEFAULT_AUDIO_OUTPUT_DIR,
    EpisodeAudioDownloadError,
    download_episode_audio,
)


PROGRAM_NAME: Final = "podcast-download-audio"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        result = download_episode_audio(
            args.episode_url,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except EpisodeAudioDownloadError as error:
        print(str(error), file=sys.stderr)
        for note in getattr(error, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("episode_url")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUDIO_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
