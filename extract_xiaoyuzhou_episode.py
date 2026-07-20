from __future__ import annotations

import sys
from typing import Final

from podcast_job_finder.xiaoyuzhou.episode_page import (
    EpisodeParseError,
    parse_episode_url,
)


USAGE_TEXT: Final = "用法：python3 extract_xiaoyuzhou_episode.py <episode_url>"


def main() -> int:
    if len(sys.argv) != 2:
        print(USAGE_TEXT, file=sys.stderr)
        return 1

    try:
        episode = parse_episode_url(sys.argv[1])
    except (EpisodeParseError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(episode.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
