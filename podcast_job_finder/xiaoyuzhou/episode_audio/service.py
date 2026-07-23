from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from podcast_job_finder.xiaoyuzhou.episode_audio.errors import (
    EpisodeAudioDownloadError,
)
from podcast_job_finder.xiaoyuzhou.episode_audio.files import (
    build_audio_target_path,
    store_episode_audio,
)
from podcast_job_finder.xiaoyuzhou.episode_client import (
    extract_episode_id_from_url,
    parse_episode_url,
)
from podcast_job_finder.xiaoyuzhou.episode_parser import EpisodeParseError


DEFAULT_AUDIO_OUTPUT_DIR: Final = Path("output/audio")
SUPPORTED_AUDIO_EXTENSIONS: Final = frozenset(
    {
        ".aac",
        ".aiff",
        ".ape",
        ".flac",
        ".m4a",
        ".mp3",
        ".mp4",
        ".ogg",
        ".wav",
        ".webm",
        ".wma",
    }
)
INVALID_EPISODE_URL_ERROR: Final = "无法从 URL 中提取小宇宙单集 ID：{url}"
MISSING_AUDIO_URL_ERROR: Final = "节目页面未提供音频 URL：{url}"
INVALID_AUDIO_URL_ERROR: Final = "节目音频 URL 无效：{url}"
UNSUPPORTED_AUDIO_EXTENSION_ERROR: Final = "不支持的节目音频扩展名：{url}"
FETCH_EPISODE_ERROR_TEMPLATE: Final = "获取节目音频信息失败：{url}，{error_message}"


@dataclass(slots=True, frozen=True)
class EpisodeAudioDownloadResult:
    eid: str
    title: str
    source_url: str
    local_path: Path
    skipped: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "eid": self.eid,
            "title": self.title,
            "source_url": self.source_url,
            "local_path": str(self.local_path),
            "skipped": self.skipped,
        }


def download_episode_audio(
    episode_url: str,
    *,
    output_dir: Path = DEFAULT_AUDIO_OUTPUT_DIR,
    overwrite: bool = False,
) -> EpisodeAudioDownloadResult:
    eid = extract_episode_id_from_url(episode_url)
    if eid is None:
        raise EpisodeAudioDownloadError(
            INVALID_EPISODE_URL_ERROR.format(url=episode_url)
        )

    try:
        episode = parse_episode_url(episode_url)
    except (EpisodeParseError, ValueError) as error:
        raise EpisodeAudioDownloadError(
            FETCH_EPISODE_ERROR_TEMPLATE.format(
                url=episode_url,
                error_message=str(error),
            )
        ) from error

    source_url = episode.audio_url
    if source_url is None:
        raise EpisodeAudioDownloadError(MISSING_AUDIO_URL_ERROR.format(url=episode_url))

    extension = _extract_audio_extension(source_url)
    target_path = build_audio_target_path(output_dir, eid, extension)
    skipped = store_episode_audio(
        source_url,
        target_path,
        overwrite=overwrite,
    )
    return EpisodeAudioDownloadResult(
        eid=eid,
        title=episode.title,
        source_url=source_url,
        local_path=target_path,
        skipped=skipped,
    )


def _extract_audio_extension(source_url: str) -> str:
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise EpisodeAudioDownloadError(INVALID_AUDIO_URL_ERROR.format(url=source_url))

    extension = Path(parsed_url.path).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise EpisodeAudioDownloadError(
            UNSUPPORTED_AUDIO_EXTENSION_ERROR.format(url=source_url)
        )
    return extension
