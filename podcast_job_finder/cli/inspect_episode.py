from __future__ import annotations

import sys
from typing import Final

from podcast_job_finder.episode import parse_episode_url
from podcast_job_finder.errors import PodcastJobFinderError


USAGE_TEXT: Final = "用法：podcast-inspect-episode <episode_url>"
HELP_FLAGS: Final = frozenset({"-h", "--help"})


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] in HELP_FLAGS:
        print(USAGE_TEXT)
        return 0
    if len(sys.argv) != 2:
        print(USAGE_TEXT, file=sys.stderr)
        return 1

    try:
        episode = parse_episode_url(sys.argv[1])
    except PodcastJobFinderError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(episode.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
