from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser
from typing import Final
from urllib.parse import urlparse

import requests

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT


NEXT_DATA_SCRIPT_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
ZERO_WIDTH_CHARACTERS_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
CONTENT_SECTION_TITLE: Final = "标题"
AUDIO_URL_SECTION_TITLE: Final = "音频 URL"
BODY_SECTION_TITLE: Final = "正文"
COMMENTS_SECTION_TITLE: Final = "评论"
NO_COMMENTS_TEXT: Final = "无评论"
TOP_LEVEL_COMMENT_TEMPLATE: Final = "评论 {index}｜作者：{author}｜时间：{created_at}"
REPLY_COMMENT_TEMPLATE: Final = "回复 {index}｜作者：{author}｜时间：{created_at}"
MISSING_NEXT_DATA_ERROR: Final = "未找到 __NEXT_DATA__ 数据块。"
INVALID_NEXT_DATA_ERROR: Final = "__NEXT_DATA__ JSON 解析失败。"
INVALID_PAGE_DATA_ERROR: Final = "__NEXT_DATA__ 页面数据格式无效。"
REQUEST_TIMEOUT_SECONDS: Final = 30
REQUEST_USER_AGENT: Final = DEFAULT_BROWSER_USER_AGENT
FETCH_URL_ERROR_TEMPLATE: Final = "请求页面失败：{url}"
INVALID_URL_ERROR_TEMPLATE: Final = "URL 无效：{url}"
DEBUG_URL_TEMPLATE: Final = "[debug] url={url}"
DEBUG_EXCEPTION_TEMPLATE: Final = (
    "[debug] exception={exception_type}: {exception_message}"
)
DEBUG_HTTP_STATUS_TEMPLATE: Final = "[debug] http_status={status_code}"
EPISODE_PATH_PREFIX: Final = "/episode/"


class EpisodeParseError(ValueError):
    """Raised when an episode page cannot be parsed."""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"br", "p", "div", "li", "ul", "ol", "section", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "div", "li", "ul", "ol", "section", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def get_text(self) -> str:
        return "".join(self._parts)


@dataclass(slots=True)
class CommentInfo:
    author: str
    created_at: str
    text: str
    replies: list["CommentInfo"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text_lines(self, index_label: str, indent_level: int = 0) -> list[str]:
        indent = "  " * indent_level
        header_template = (
            TOP_LEVEL_COMMENT_TEMPLATE if indent_level == 0 else REPLY_COMMENT_TEMPLATE
        )
        header_text = header_template.format(
            index=index_label,
            author=self.author,
            created_at=self.created_at,
        )
        lines = [
            f"{indent}{header_text}",
            f"{indent}{self.text}",
        ]
        for reply_index, reply in enumerate(self.replies, start=1):
            reply_label = f"{index_label}.{reply_index}"
            lines.extend(reply.to_text_lines(reply_label, indent_level + 1))
        return lines


@dataclass(slots=True)
class EpisodeInfo:
    title: str
    content: str
    audio_url: str | None = None
    comments: list[CommentInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        sections = [
            CONTENT_SECTION_TITLE,
            self.title,
            "",
            AUDIO_URL_SECTION_TITLE,
            self.audio_url or "",
            "",
            BODY_SECTION_TITLE,
            self.content,
            "",
            COMMENTS_SECTION_TITLE,
        ]
        if not self.comments:
            sections.append(NO_COMMENTS_TEXT)
            return "\n".join(sections)

        comment_lines: list[str] = []
        for comment_index, comment in enumerate(self.comments, start=1):
            comment_lines.extend(comment.to_text_lines(str(comment_index)))
        sections.append("\n".join(comment_lines))
        return "\n".join(sections)


def parse_episode_html(html_text: str) -> EpisodeInfo:
    page_props = _extract_page_props(html_text)
    episode_data = _require_dict(page_props.get("episode"))
    comments_data = _require_optional_list(page_props.get("comments"))

    title = _clean_text(str(episode_data.get("title") or ""))
    content = _extract_episode_content(episode_data)
    comments = [_parse_comment(comment_data) for comment_data in comments_data]
    return EpisodeInfo(
        title=title,
        content=content,
        audio_url=_extract_audio_url(episode_data),
        comments=comments,
    )


def parse_episode_url(episode_url: str) -> EpisodeInfo:
    return parse_episode_html(fetch_episode_html(episode_url))


def extract_episode_audio_url(episode_url: str) -> str | None:
    return parse_episode_url(episode_url).audio_url


def extract_episode_id_from_url(episode_url: str) -> str | None:
    if not episode_url.startswith(("http://", "https://")):
        return None

    parsed_url = urlparse(episode_url)
    normalized_path = parsed_url.path.rstrip("/")
    if not normalized_path.startswith(EPISODE_PATH_PREFIX):
        return None

    episode_id = normalized_path.removeprefix(EPISODE_PATH_PREFIX).strip()
    if not episode_id or "/" in episode_id:
        return None
    return episode_id


def fetch_episode_html(episode_url: str) -> str:
    if not episode_url.startswith(("http://", "https://")):
        raise ValueError(INVALID_URL_ERROR_TEMPLATE.format(url=episode_url))

    request_headers = {"User-Agent": REQUEST_USER_AGENT}
    try:
        response = requests.get(
            episode_url,
            headers=request_headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        return response.text
    except requests.RequestException as exc:
        raise ValueError(_build_fetch_error_message(episode_url, exc)) from exc


def _extract_page_props(html_text: str) -> dict[str, object]:
    matched_script = NEXT_DATA_SCRIPT_PATTERN.search(html_text)
    if matched_script is None:
        raise EpisodeParseError(MISSING_NEXT_DATA_ERROR)

    try:
        parsed_next_data: object = json.loads(matched_script.group(1))
    except json.JSONDecodeError as error:
        raise EpisodeParseError(INVALID_NEXT_DATA_ERROR) from error

    next_data = _require_dict(parsed_next_data)
    props = _require_dict(next_data.get("props"))
    return _require_dict(props.get("pageProps"))


def _build_fetch_error_message(episode_url: str, exc: Exception) -> str:
    debug_lines = [
        FETCH_URL_ERROR_TEMPLATE.format(url=episode_url),
        DEBUG_URL_TEMPLATE.format(url=episode_url),
        DEBUG_EXCEPTION_TEMPLATE.format(
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        ),
    ]

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        debug_lines.append(DEBUG_HTTP_STATUS_TEMPLATE.format(status_code=status_code))

    return "\n".join(debug_lines)


def _extract_episode_content(episode_data: dict[str, object]) -> str:
    description = _clean_text(str(episode_data.get("description") or ""))
    if description:
        return description

    shownotes = str(episode_data.get("shownotes") or "")
    return _clean_text(_strip_html_tags(shownotes))


def _extract_audio_url(episode_data: dict[str, object]) -> str | None:
    enclosure_data = episode_data.get("enclosure")
    if not isinstance(enclosure_data, dict):
        return None

    audio_url = enclosure_data.get("url")
    if not isinstance(audio_url, str):
        return None

    return audio_url.strip() or None


def _parse_comment(comment_data: object) -> CommentInfo:
    comment_payload = _require_dict(comment_data)
    author_data = _require_optional_dict(comment_payload.get("author"))
    replies_data = _require_optional_list(comment_payload.get("replies"))
    return CommentInfo(
        author=_clean_text(str(author_data.get("nickname") or "")),
        created_at=_clean_text(str(comment_payload.get("createdAt") or "")),
        text=_clean_text(str(comment_payload.get("text") or "")),
        replies=[_parse_comment(reply_data) for reply_data in replies_data],
    )


def _require_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EpisodeParseError(INVALID_PAGE_DATA_ERROR)
    return value


def _require_optional_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    return _require_dict(value)


def _require_optional_list(value: object) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EpisodeParseError(INVALID_PAGE_DATA_ERROR)
    return value


def _strip_html_tags(html_fragment: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html_fragment)
    extractor.close()
    return unescape(extractor.get_text())


def _clean_text(text: str) -> str:
    cleaned_text = ZERO_WIDTH_CHARACTERS_PATTERN.sub("", text).replace("\r\n", "\n")
    cleaned_text = cleaned_text.replace("\r", "\n")
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()
