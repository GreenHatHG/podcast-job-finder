from __future__ import annotations

from typing import Final

from podcast_job_finder.xiaoyuzhou.xyz.auth_store import (
    XiaoyuzhouAuthSession,
    save_auth_session,
    update_auth_session_tokens,
)
from podcast_job_finder.xiaoyuzhou.xyz.client import (
    XyzClient,
    XyzUnauthorizedError,
)
from podcast_job_finder.xiaoyuzhou.xyz.models import (
    EpisodeListPage,
    EpisodeLoadMoreKey,
    PodcastEpisodeSummary,
)


DEFAULT_EPISODE_ORDER: Final = "desc"


def list_podcast_episodes(
    *,
    xyz_client: XyzClient,
    auth_session: XiaoyuzhouAuthSession,
    pid: str,
    fetch_all: bool,
) -> tuple[PodcastEpisodeSummary, ...]:
    episodes: list[PodcastEpisodeSummary] = []
    load_more_key: EpisodeLoadMoreKey | None = None
    current_session = auth_session
    while True:
        page, current_session = _fetch_page_with_refresh(
            xyz_client=xyz_client,
            auth_session=current_session,
            pid=pid,
            load_more_key=load_more_key,
        )
        episodes.extend(page.episodes)
        if not fetch_all or page.load_more_key is None:
            return tuple(episodes)
        load_more_key = page.load_more_key


def _fetch_page_with_refresh(
    *,
    xyz_client: XyzClient,
    auth_session: XiaoyuzhouAuthSession,
    pid: str,
    load_more_key: EpisodeLoadMoreKey | None,
) -> tuple[EpisodeListPage, XiaoyuzhouAuthSession]:
    try:
        page = xyz_client.list_podcast_episodes(
            pid=pid,
            access_token=auth_session.access_token,
            load_more_key=load_more_key,
            order=DEFAULT_EPISODE_ORDER,
        )
        return page, auth_session
    except XyzUnauthorizedError:
        refreshed_session = _refresh_auth_session(xyz_client, auth_session)
        page = xyz_client.list_podcast_episodes(
            pid=pid,
            access_token=refreshed_session.access_token,
            load_more_key=load_more_key,
            order=DEFAULT_EPISODE_ORDER,
        )
        return page, refreshed_session


def _refresh_auth_session(
    xyz_client: XyzClient,
    auth_session: XiaoyuzhouAuthSession,
) -> XiaoyuzhouAuthSession:
    refreshed_tokens = xyz_client.refresh_token(
        access_token=auth_session.access_token,
        refresh_token=auth_session.refresh_token,
    )
    refreshed_session = update_auth_session_tokens(
        auth_session,
        access_token=refreshed_tokens.access_token,
        refresh_token=refreshed_tokens.refresh_token,
    )
    save_auth_session(refreshed_session)
    return refreshed_session
