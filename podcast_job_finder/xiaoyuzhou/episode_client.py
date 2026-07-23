from __future__ import annotations

import re
from typing import Final
from urllib.parse import urlparse

import requests

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT
from podcast_job_finder.xiaoyuzhou.episode_parser import parse_episode_html
from podcast_job_finder.xiaoyuzhou.models import EpisodeInfo


EPISODE_ID_PATTERN = re.compile(r"^[0-9A-Za-z]{24}$")
REQUEST_TIMEOUT_SECONDS: Final = 30
FETCH_URL_ERROR_TEMPLATE: Final = "请求页面失败：{url}"
INVALID_URL_ERROR_TEMPLATE: Final = "URL 无效：{url}"
DEBUG_URL_TEMPLATE: Final = "[debug] url={url}"
DEBUG_EXCEPTION_TEMPLATE: Final = (
    "[debug] exception={exception_type}: {exception_message}"
)
DEBUG_HTTP_STATUS_TEMPLATE: Final = "[debug] http_status={status_code}"
EPISODE_PATH_PREFIX: Final = "/episode/"
EPISODE_URL_TEMPLATE: Final = "https://www.xiaoyuzhoufm.com/episode/{eid}"


def parse_episode_url(episode_url: str) -> EpisodeInfo:
    return parse_episode_html(fetch_episode_html(episode_url))


def extract_episode_audio_url(episode_url: str) -> str | None:
    return parse_episode_url(episode_url).audio_url


def extract_episode_id_from_url(episode_url: str) -> str | None:
    if not episode_url.startswith(("http://", "https://")):
        return None

    normalized_path = urlparse(episode_url).path.rstrip("/")
    if not normalized_path.startswith(EPISODE_PATH_PREFIX):
        return None

    episode_id = normalized_path.removeprefix(EPISODE_PATH_PREFIX).strip()
    if EPISODE_ID_PATTERN.fullmatch(episode_id) is None:
        return None
    return episode_id


def build_episode_url(eid: str) -> str:
    return EPISODE_URL_TEMPLATE.format(eid=eid)


def fetch_episode_html(episode_url: str) -> str:
    if not episode_url.startswith(("http://", "https://")):
        raise ValueError(INVALID_URL_ERROR_TEMPLATE.format(url=episode_url))
    try:
        response = requests.get(
            episode_url,
            headers={"User-Agent": DEFAULT_BROWSER_USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    except requests.RequestException as error:
        raise ValueError(_build_fetch_error_message(episode_url, error)) from error


def _build_fetch_error_message(episode_url: str, error: Exception) -> str:
    debug_lines = [
        FETCH_URL_ERROR_TEMPLATE.format(url=episode_url),
        DEBUG_URL_TEMPLATE.format(url=episode_url),
        DEBUG_EXCEPTION_TEMPLATE.format(
            exception_type=type(error).__name__,
            exception_message=str(error),
        ),
    ]
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        debug_lines.append(DEBUG_HTTP_STATUS_TEMPLATE.format(status_code=status_code))
    return "\n".join(debug_lines)
