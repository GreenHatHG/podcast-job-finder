from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from podcast_job_finder.errors import ConfigurationError


DEFAULT_PODCAST_CATALOG_PATH: Final = Path("podcasts.toml")
PODCASTS_TABLE_NAME: Final = "podcasts"


def resolve_feed_reference(
    reference: str,
    *,
    catalog_path: Path = DEFAULT_PODCAST_CATALOG_PATH,
) -> str:
    """把 HTTP/HTTPS 地址原样返回，或把播客名称解析为 RSS 地址。"""
    normalized_reference = reference.strip()
    if not normalized_reference:
        raise ConfigurationError("播客名称或 RSS 地址不能为空。")

    if normalized_reference.lower().startswith(("http://", "https://")):
        if not _is_valid_feed_url(normalized_reference):
            raise ConfigurationError(f"RSS 地址无效：{normalized_reference}")
        return normalized_reference

    return get_podcast_feed_url(normalized_reference, catalog_path=catalog_path)


def get_podcast_feed_url(
    podcast_name: str,
    *,
    catalog_path: Path = DEFAULT_PODCAST_CATALOG_PATH,
) -> str:
    normalized_name = podcast_name.strip()
    if not normalized_name:
        raise ConfigurationError("播客名称不能为空。")

    feed_urls = _load_podcast_feed_urls(catalog_path)
    try:
        return feed_urls[normalized_name]
    except KeyError as error:
        available_names = "、".join(feed_urls)
        raise ConfigurationError(
            f"{catalog_path} 中未找到播客“{normalized_name}”。"
            f"可用名称：{available_names}"
        ) from error


def _load_podcast_feed_urls(catalog_path: Path) -> dict[str, str]:
    catalog_data = _read_catalog_file(catalog_path)
    podcasts = catalog_data.get(PODCASTS_TABLE_NAME)
    if not isinstance(podcasts, dict):
        raise ConfigurationError(
            f"播客配置文件缺少 [{PODCASTS_TABLE_NAME}]：{catalog_path}"
        )
    if not podcasts:
        raise ConfigurationError(f"播客配置文件中没有播客：{catalog_path}")

    feed_urls: dict[str, str] = {}
    for raw_name, raw_feed_url in podcasts.items():
        podcast_name = raw_name.strip()
        if not podcast_name:
            raise ConfigurationError(f"播客配置中存在空名称：{catalog_path}")
        if podcast_name in feed_urls:
            raise ConfigurationError(
                f"播客配置中存在重复名称“{podcast_name}”：{catalog_path}"
            )
        if not isinstance(raw_feed_url, str):
            raise ConfigurationError(
                f"播客“{podcast_name}”的 RSS 地址必须是字符串：{catalog_path}"
            )

        feed_url = raw_feed_url.strip()
        if not _is_valid_feed_url(feed_url):
            raise ConfigurationError(f"播客“{podcast_name}”的 RSS 地址无效：{feed_url}")
        feed_urls[podcast_name] = feed_url

    return feed_urls


def _is_valid_feed_url(feed_url: str) -> bool:
    try:
        parsed_url = urlparse(feed_url)
    except ValueError:
        return False
    return parsed_url.scheme.lower() in {"http", "https"} and bool(parsed_url.netloc)


def _read_catalog_file(catalog_path: Path) -> dict[str, object]:
    try:
        with catalog_path.open("rb") as catalog_file:
            return tomllib.load(catalog_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"未找到播客配置文件：{catalog_path}") from error
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ConfigurationError(
            f"解析播客配置文件失败：{catalog_path}，{error}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"读取播客配置文件失败：{catalog_path}，{error}"
        ) from error
