from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from html import unescape
from html.parser import HTMLParser

import requests

from http_user_agents import DEFAULT_BROWSER_USER_AGENT


NEXT_DATA_SCRIPT_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
ZERO_WIDTH_CHARACTERS_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
CONTENT_SECTION_TITLE = "标题"
BODY_SECTION_TITLE = "正文"
COMMENTS_SECTION_TITLE = "评论"
NO_COMMENTS_TEXT = "无评论"
TOP_LEVEL_COMMENT_TEMPLATE = "评论 {index}｜作者：{author}｜时间：{created_at}"
REPLY_COMMENT_TEMPLATE = "回复 {index}｜作者：{author}｜时间：{created_at}"
MISSING_NEXT_DATA_ERROR = "未找到 __NEXT_DATA__ 数据块。"
INVALID_NEXT_DATA_ERROR = "__NEXT_DATA__ JSON 解析失败。"
USAGE_TEXT = "用法：python3 extract_xiaoyuzhou_episode.py <episode_url>"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_USER_AGENT = DEFAULT_BROWSER_USER_AGENT
FETCH_URL_ERROR_TEMPLATE = "请求页面失败：{url}"
INVALID_URL_ERROR_TEMPLATE = "URL 无效：{url}"
DEBUG_LOG_PREFIX = "[debug]"
DEBUG_URL_TEMPLATE = "[debug] url={url}"
DEBUG_EXCEPTION_TEMPLATE = "[debug] exception={exception_type}: {exception_message}"
DEBUG_HTTP_STATUS_TEMPLATE = "[debug] http_status={status_code}"


class EpisodeParseError(ValueError):
    """Raised when an episode page cannot be parsed."""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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
        lines = [
            f"{indent}{header_template.format(index=index_label, author=self.author, created_at=self.created_at)}",
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
    comments: list[CommentInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        sections = [
            CONTENT_SECTION_TITLE,
            self.title,
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
    episode_data = page_props.get("episode") or {}
    comments_data = page_props.get("comments") or []

    title = _clean_text(str(episode_data.get("title") or ""))
    content = _extract_episode_content(episode_data)
    comments = [_parse_comment(comment_data) for comment_data in comments_data]
    return EpisodeInfo(title=title, content=content, comments=comments)


def parse_episode_url(episode_url: str) -> EpisodeInfo:
    return parse_episode_html(fetch_episode_html(episode_url))


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


def main() -> int:
    if len(sys.argv) != 2:
        print(USAGE_TEXT, file=sys.stderr)
        return 1

    episode_url = sys.argv[1]
    try:
        episode = parse_episode_url(episode_url)
    except (EpisodeParseError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(episode.to_text())
    return 0


def _extract_page_props(html_text: str) -> dict:
    matched_script = NEXT_DATA_SCRIPT_PATTERN.search(html_text)
    if matched_script is None:
        raise EpisodeParseError(MISSING_NEXT_DATA_ERROR)

    try:
        next_data = json.loads(matched_script.group(1))
    except json.JSONDecodeError as error:
        raise EpisodeParseError(INVALID_NEXT_DATA_ERROR) from error

    return next_data.get("props", {}).get("pageProps", {})


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


def _extract_episode_content(episode_data: dict) -> str:
    description = _clean_text(str(episode_data.get("description") or ""))
    if description:
        return description

    shownotes = str(episode_data.get("shownotes") or "")
    return _clean_text(_strip_html_tags(shownotes))


def _parse_comment(comment_data: dict) -> CommentInfo:
    author_data = comment_data.get("author") or {}
    replies_data = comment_data.get("replies") or []
    return CommentInfo(
        author=_clean_text(str(author_data.get("nickname") or "")),
        created_at=_clean_text(str(comment_data.get("createdAt") or "")),
        text=_clean_text(str(comment_data.get("text") or "")),
        replies=[_parse_comment(reply_data) for reply_data in replies_data],
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
