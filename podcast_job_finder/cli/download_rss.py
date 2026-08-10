from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

from podcast_job_finder.logging import configure_logging
from podcast_job_finder.rss.download import (
    DEFAULT_RSS_AUDIO_OUTPUT_DIR,
    DEFAULT_RSS_OUTPUT_DIR,
    download_rss_feed,
)
from podcast_job_finder.rss.feed import RssFeedError


PROGRAM_NAME: Final = "podcast-download-rss"


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    results: list[dict[str, object]] = []
    has_failure = False

    for feed_url in args.feed_urls:
        try:
            result = download_rss_feed(
                feed_url,
                output_dir=args.output_dir,
                audio_output_dir=args.audio_output_dir,
                overwrite=args.overwrite,
                list_only=args.list_only,
            )
        except RssFeedError as error:
            has_failure = True
            results.append({"feed_url": feed_url, "error": str(error)})
            print(str(error), file=sys.stderr)
            continue
        result_data = result.to_dict()
        results.append(result_data)
        if result.failed_count > 0:
            has_failure = True

    print(json.dumps({"feeds": results}, ensure_ascii=False, indent=2))
    return 1 if has_failure else 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM_NAME)
    parser.add_argument("feed_urls", nargs="+")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RSS_OUTPUT_DIR,
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只保存完整节目清单，不下载音频。",
    )
    parser.add_argument(
        "--audio-output-dir",
        type=Path,
        default=DEFAULT_RSS_AUDIO_OUTPUT_DIR,
        help="音频保存目录，默认复用 output/audio。",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
