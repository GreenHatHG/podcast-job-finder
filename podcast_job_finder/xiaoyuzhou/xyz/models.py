from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LoginResult:
    uid: str
    access_token: str
    refresh_token: str


@dataclass(slots=True, frozen=True)
class RefreshedTokens:
    access_token: str
    refresh_token: str


@dataclass(slots=True, frozen=True)
class EpisodeLoadMoreKey:
    pub_date: str
    id: str
    direction: str


@dataclass(slots=True, frozen=True)
class PodcastEpisodeSummary:
    pid: str
    eid: str
    title: str
    pub_date: str


@dataclass(slots=True, frozen=True)
class EpisodeListPage:
    episodes: tuple[PodcastEpisodeSummary, ...]
    load_more_key: EpisodeLoadMoreKey | None
