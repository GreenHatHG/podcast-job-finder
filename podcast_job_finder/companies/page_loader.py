from __future__ import annotations

from typing import Final, Protocol

from podcast_job_finder.llm.rate_limit import PerMinuteRateLimiter
from podcast_job_finder.xiaoyuzhou.episode_client import parse_episode_url
from podcast_job_finder.xiaoyuzhou.models import EpisodeInfo


class EpisodePageLoaderProtocol(Protocol):
    def load(self, episode_url: str) -> EpisodeInfo: ...


class EpisodePageLoader:
    def load(self, episode_url: str) -> EpisodeInfo:
        return parse_episode_url(episode_url)


class RateLimitedEpisodePageLoader:
    def __init__(
        self,
        wrapped_loader: EpisodePageLoaderProtocol,
        rate_per_minute: float | None,
    ) -> None:
        self._wrapped_loader = wrapped_loader
        self._rate_limiter = PerMinuteRateLimiter(rate_per_minute)

    def load(self, episode_url: str) -> EpisodeInfo:
        self._rate_limiter.wait_turn()
        return self._wrapped_loader.load(episode_url)


DEFAULT_EPISODE_PAGE_LOADER: Final = EpisodePageLoader()
