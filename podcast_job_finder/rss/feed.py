from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Final
from xml.etree import ElementTree

import requests

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT
from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioDownloadError
from podcast_job_finder.audio.episode_audio.service import extract_audio_extension
from podcast_job_finder.errors import PodcastJobFinderError


logger = logging.getLogger(__name__)

FEED_CONNECT_TIMEOUT_SECONDS: Final = 10
FEED_READ_TIMEOUT_SECONDS: Final = 60
MAX_EPISODE_ID_LENGTH: Final = 120
FALLBACK_ID_HASH_LENGTH: Final = 24
FEED_ID_HASH_LENGTH: Final = 16
SAFE_EPISODE_ID: Final = re.compile(r"^[A-Za-z0-9._-]+$")
FETCH_FEED_ERROR_TEMPLATE: Final = "获取播客 RSS 失败：{url}，{error_message}"
PARSE_FEED_ERROR_TEMPLATE: Final = "解析播客 RSS 失败：{url}，{error_message}"
MISSING_CHANNEL_ERROR: Final = "RSS 中未找到 channel：{url}"
MISSING_FEED_TITLE_ERROR: Final = "RSS 中未找到播客名称：{url}"
MISSING_EPISODE_TITLE_ERROR: Final = "RSS 第 {index} 个节目缺少标题：{url}"
MISSING_AUDIO_URL_ERROR: Final = "RSS 节目缺少音频地址：title={title} feed={url}"
INVALID_AUDIO_URL_ERROR_TEMPLATE: Final = (
    "RSS 节目音频地址无效：title={title} feed={url}，{error_message}"
)
DUPLICATE_EPISODE_ID_ERROR: Final = (
    "RSS 中存在重复节目 ID：episode_id={episode_id} feed={url}"
)


class RssFeedError(PodcastJobFinderError, RuntimeError):
    """读取或解析播客 RSS 时发生错误。"""


# RSS 单集需要同时保留节目标识和音频资源的原始字段。
# pylint: disable=too-many-instance-attributes
@dataclass(slots=True, frozen=True)
class RssEpisode:
    episode_id: str
    guid: str
    title: str
    link: str | None
    published_at: str | None
    duration: str | None
    audio_url: str
    audio_type: str | None
    audio_length_bytes: int | None
    extension: str


# pylint: enable=too-many-instance-attributes


@dataclass(slots=True, frozen=True)
class RssFeed:
    source_url: str
    title: str
    episodes: tuple[RssEpisode, ...]

    @property
    def feed_id(self) -> str:
        return hashlib.sha256(self.source_url.encode()).hexdigest()[
            :FEED_ID_HASH_LENGTH
        ]


def fetch_rss_feed(feed_url: str) -> RssFeed:
    try:
        response = requests.get(
            feed_url,
            headers={"User-Agent": DEFAULT_BROWSER_USER_AGENT},
            timeout=(FEED_CONNECT_TIMEOUT_SECONDS, FEED_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
    except requests.RequestException as error:
        raise RssFeedError(
            FETCH_FEED_ERROR_TEMPLATE.format(
                url=feed_url,
                error_message=str(error),
            )
        ) from error

    try:
        root = ElementTree.fromstring(response.content)
    except ElementTree.ParseError as error:
        raise RssFeedError(
            PARSE_FEED_ERROR_TEMPLATE.format(
                url=feed_url,
                error_message=str(error),
            )
        ) from error
    return _parse_rss_root(root, feed_url)


def _parse_rss_root(root: ElementTree.Element, feed_url: str) -> RssFeed:
    channel = _find_child(root, "channel")
    if channel is None:
        raise RssFeedError(MISSING_CHANNEL_ERROR.format(url=feed_url))
    feed_title = _child_text(channel, "title")
    if feed_title is None:
        raise RssFeedError(MISSING_FEED_TITLE_ERROR.format(url=feed_url))

    episodes: list[RssEpisode] = []
    episode_ids: set[str] = set()
    for index, item in enumerate(_find_children(channel, "item"), start=1):
        episode = _parse_episode(item, feed_url, index)
        if episode.episode_id in episode_ids:
            raise RssFeedError(
                DUPLICATE_EPISODE_ID_ERROR.format(
                    episode_id=episode.episode_id,
                    url=feed_url,
                )
            )
        episode_ids.add(episode.episode_id)
        episodes.append(episode)

    logger.info(
        "读取播客 RSS 完成：podcast=%s episodes=%d feed=%s",
        feed_title,
        len(episodes),
        feed_url,
    )
    return RssFeed(source_url=feed_url, title=feed_title, episodes=tuple(episodes))


def _parse_episode(
    item: ElementTree.Element,
    feed_url: str,
    index: int,
) -> RssEpisode:
    title = _child_text(item, "title")
    if title is None:
        raise RssFeedError(
            MISSING_EPISODE_TITLE_ERROR.format(index=index, url=feed_url)
        )
    enclosure = _find_child(item, "enclosure")
    audio_url = enclosure.get("url", "").strip() if enclosure is not None else ""
    if not audio_url:
        raise RssFeedError(MISSING_AUDIO_URL_ERROR.format(title=title, url=feed_url))
    try:
        extension = extract_audio_extension(audio_url)
    except EpisodeAudioDownloadError as error:
        raise RssFeedError(
            INVALID_AUDIO_URL_ERROR_TEMPLATE.format(
                title=title,
                url=feed_url,
                error_message=str(error),
            )
        ) from error

    guid = _child_text(item, "guid") or _child_text(item, "link") or audio_url
    audio_length = enclosure.get("length") if enclosure is not None else None
    return RssEpisode(
        episode_id=_build_episode_id(guid),
        guid=guid,
        title=title,
        link=_child_text(item, "link"),
        published_at=_child_text(item, "pubDate"),
        duration=_child_text(item, "duration"),
        audio_url=audio_url,
        audio_type=_optional_attribute(enclosure, "type"),
        audio_length_bytes=_parse_optional_non_negative_int(audio_length),
        extension=extension,
    )


def _build_episode_id(guid: str) -> str:
    if len(guid) <= MAX_EPISODE_ID_LENGTH and SAFE_EPISODE_ID.fullmatch(guid):
        return guid
    return hashlib.sha256(guid.encode()).hexdigest()[:FALLBACK_ID_HASH_LENGTH]


def _find_child(
    element: ElementTree.Element,
    local_name: str,
) -> ElementTree.Element | None:
    return next(
        (child for child in element if _local_name(child.tag) == local_name),
        None,
    )


def _find_children(
    element: ElementTree.Element,
    local_name: str,
) -> list[ElementTree.Element]:
    return [child for child in element if _local_name(child.tag) == local_name]


def _child_text(element: ElementTree.Element, local_name: str) -> str | None:
    child = _find_child(element, local_name)
    if child is None or child.text is None:
        return None
    normalized_text = child.text.strip()
    return normalized_text or None


def _local_name(tag: str) -> str:
    return tag.rpartition("}")[2]


def _optional_attribute(
    element: ElementTree.Element | None,
    name: str,
) -> str | None:
    if element is None:
        return None
    value = element.get(name, "").strip()
    return value or None


def _parse_optional_non_negative_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed_value = int(value)
    except ValueError:
        return None
    return parsed_value if parsed_value >= 0 else None
