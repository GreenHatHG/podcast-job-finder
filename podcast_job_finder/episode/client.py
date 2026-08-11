from __future__ import annotations

from typing import Final

import requests

from podcast_job_finder.episode.models import EpisodeInfo
from podcast_job_finder.episode.parser import parse_episode_html
from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT


REQUEST_TIMEOUT_SECONDS: Final = 30
FETCH_URL_ERROR_TEMPLATE: Final = "请求页面失败：{url}"
INVALID_URL_ERROR_TEMPLATE: Final = "URL 无效：{url}"
DEBUG_URL_TEMPLATE: Final = "[debug] url={url}"
DEBUG_EXCEPTION_TEMPLATE: Final = (
    "[debug] exception={exception_type}: {exception_message}"
)
DEBUG_HTTP_STATUS_TEMPLATE: Final = "[debug] http_status={status_code}"


def parse_episode_url(episode_url: str) -> EpisodeInfo:
    return parse_episode_html(fetch_episode_html(episode_url))


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
