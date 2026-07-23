from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from typing import Final

from podcast_job_finder.xiaoyuzhou.models import CommentInfo, EpisodeInfo


NEXT_DATA_SCRIPT_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
ZERO_WIDTH_CHARACTERS_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")
MISSING_NEXT_DATA_ERROR: Final = "未找到 __NEXT_DATA__ 数据块。"
INVALID_NEXT_DATA_ERROR: Final = "__NEXT_DATA__ JSON 解析失败。"
INVALID_PAGE_DATA_ERROR_TEMPLATE: Final = (
    "__NEXT_DATA__ 字段格式无效：{field_path}，期望{expected_type}，"
    "实际为{actual_type}。"
)
MISSING_PAGE_DATA_FIELD_ERROR_TEMPLATE: Final = (
    "__NEXT_DATA__ 缺少必要字段：{field_path}。"
)
JSON_OBJECT_TYPE_DESCRIPTION: Final = "对象"
JSON_LIST_TYPE_DESCRIPTION: Final = "列表"
JSON_STRING_TYPE_DESCRIPTION: Final = "字符串"
MISSING_JSON_FIELD: Final = object()
JSON_TYPE_DESCRIPTIONS: Final[dict[type[object], str]] = {
    type(None): "null",
    bool: "布尔值",
    dict: JSON_OBJECT_TYPE_DESCRIPTION,
    list: JSON_LIST_TYPE_DESCRIPTION,
    str: JSON_STRING_TYPE_DESCRIPTION,
    int: "整数",
    float: "数字",
}


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


def parse_episode_html(html_text: str) -> EpisodeInfo:
    page_props = _extract_page_props(html_text)
    episode_data = _require_dict(
        _require_field(page_props, "episode", "props.pageProps.episode"),
        "props.pageProps.episode",
    )
    comments_field_path = "props.pageProps.comments"
    comments_data = _require_optional_list(
        page_props.get("comments"),
        comments_field_path,
    )
    return EpisodeInfo(
        title=_clean_text(str(episode_data.get("title") or "")),
        content=_extract_episode_content(episode_data),
        audio_url=_extract_audio_url(episode_data),
        comments=[
            _parse_comment(comment_data, f"{comments_field_path}[{comment_index}]")
            for comment_index, comment_data in enumerate(comments_data)
        ],
    )


def _extract_page_props(html_text: str) -> dict[str, object]:
    matched_script = NEXT_DATA_SCRIPT_PATTERN.search(html_text)
    if matched_script is None:
        raise EpisodeParseError(MISSING_NEXT_DATA_ERROR)
    try:
        parsed_next_data: object = json.loads(matched_script.group(1))
    except json.JSONDecodeError as error:
        raise EpisodeParseError(INVALID_NEXT_DATA_ERROR) from error

    next_data = _require_dict(parsed_next_data, "__NEXT_DATA__")
    props = _require_dict(_require_field(next_data, "props", "props"), "props")
    return _require_dict(
        _require_field(props, "pageProps", "props.pageProps"),
        "props.pageProps",
    )


def _extract_episode_content(episode_data: dict[str, object]) -> str:
    description = _clean_text(str(episode_data.get("description") or ""))
    if description:
        return description
    return _clean_text(_strip_html_tags(str(episode_data.get("shownotes") or "")))


def _extract_audio_url(episode_data: dict[str, object]) -> str | None:
    enclosure_field_path = "props.pageProps.episode.enclosure"
    enclosure_value = episode_data.get("enclosure", MISSING_JSON_FIELD)
    if enclosure_value is MISSING_JSON_FIELD or enclosure_value is None:
        return None

    enclosure_data = _require_dict(enclosure_value, enclosure_field_path)
    audio_url_field_path = f"{enclosure_field_path}.url"
    audio_url = enclosure_data.get("url", MISSING_JSON_FIELD)
    if audio_url is MISSING_JSON_FIELD or audio_url is None:
        return None
    if not isinstance(audio_url, str):
        raise _build_page_data_error(
            field_path=audio_url_field_path,
            expected_type=JSON_STRING_TYPE_DESCRIPTION,
            actual_value=audio_url,
        )
    return audio_url.strip() or None


def _parse_comment(comment_data: object, field_path: str) -> CommentInfo:
    comment_payload = _require_dict(comment_data, field_path)
    author_data = _require_optional_dict(
        comment_payload.get("author"),
        f"{field_path}.author",
    )
    replies_field_path = f"{field_path}.replies"
    replies_data = _require_optional_list(
        comment_payload.get("replies"),
        replies_field_path,
    )
    return CommentInfo(
        author=_clean_text(str(author_data.get("nickname") or "")),
        created_at=_clean_text(str(comment_payload.get("createdAt") or "")),
        text=_clean_text(str(comment_payload.get("text") or "")),
        replies=[
            _parse_comment(reply_data, f"{replies_field_path}[{reply_index}]")
            for reply_index, reply_data in enumerate(replies_data)
        ],
    )


def _require_dict(value: object, field_path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _build_page_data_error(
            field_path=field_path,
            expected_type=JSON_OBJECT_TYPE_DESCRIPTION,
            actual_value=value,
        )
    return value


def _require_field(
    payload: dict[str, object],
    field_name: str,
    field_path: str,
) -> object:
    value = payload.get(field_name, MISSING_JSON_FIELD)
    if value is MISSING_JSON_FIELD:
        raise EpisodeParseError(
            MISSING_PAGE_DATA_FIELD_ERROR_TEMPLATE.format(field_path=field_path)
        )
    return value


def _require_optional_dict(value: object, field_path: str) -> dict[str, object]:
    if value is None:
        return {}
    return _require_dict(value, field_path)


def _require_optional_list(value: object, field_path: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _build_page_data_error(
            field_path=field_path,
            expected_type=JSON_LIST_TYPE_DESCRIPTION,
            actual_value=value,
        )
    return value


def _build_page_data_error(
    *,
    field_path: str,
    expected_type: str,
    actual_value: object,
) -> EpisodeParseError:
    return EpisodeParseError(
        INVALID_PAGE_DATA_ERROR_TEMPLATE.format(
            field_path=field_path,
            expected_type=expected_type,
            actual_type=_describe_json_type(actual_value),
        )
    )


def _describe_json_type(value: object) -> str:
    value_type = type(value)
    return JSON_TYPE_DESCRIPTIONS.get(value_type, value_type.__name__)


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
