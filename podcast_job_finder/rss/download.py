from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from podcast_job_finder.filesystem import (
    DEFAULT_FILE_CREATION_MODE,
    atomic_write_json,
)
from podcast_job_finder.rss.feed import (
    RssEpisode,
    RssFeed,
    RssFeedError,
    fetch_rss_feed,
)
from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioDownloadError
from podcast_job_finder.audio.episode_audio.files import (
    SOURCE_FILE_STEM,
    prepare_episode_audio_directory,
    store_episode_audio,
)
from podcast_job_finder.output_paths import (
    EPISODE_AUDIO_DIR_NAME,
    EPISODE_OUTPUT_DIR,
    FEED_OUTPUT_DIR,
    build_named_directory_name,
    find_episode_output_dir,
)


logger = logging.getLogger(__name__)

MANIFEST_FILE_NAME: Final = "manifest.json"
FEED_HASH_LENGTH: Final = 8
PREPARE_PODCAST_DIR_ERROR_TEMPLATE: Final = (
    "创建播客输出目录失败：{path}，{error_message}"
)
PODCAST_DIR_SYMLINK_ERROR: Final = "播客输出目录是符号链接，已拒绝操作：{path}"
PODCAST_DIR_REDIRECT_ERROR_TEMPLATE: Final = (
    "播客输出目录真实位置异常：期望 {expected_path}，实际 {actual_path}"
)


@dataclass(slots=True)
class EpisodeDownloadEntry:
    episode: RssEpisode
    local_path: Path
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode.episode_id,
            "guid": self.episode.guid,
            "title": self.episode.title,
            "link": self.episode.link,
            "published_at": self.episode.published_at,
            "duration": self.episode.duration,
            "audio_url": self.episode.audio_url,
            "audio_type": self.episode.audio_type,
            "audio_length_bytes": self.episode.audio_length_bytes,
            "local_path": str(self.local_path),
            "download_status": self.status,
            "error": self.error,
        }


@dataclass(slots=True, frozen=True)
class RssDownloadResult:
    feed_url: str
    podcast_title: str
    manifest_path: Path
    episode_count: int
    downloaded_count: int
    skipped_count: int
    failed_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "feed_url": self.feed_url,
            "podcast_title": self.podcast_title,
            "manifest_path": str(self.manifest_path),
            "episode_count": self.episode_count,
            "downloaded_count": self.downloaded_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
        }


def download_rss_feed(
    feed_url: str,
    *,
    output_dir: Path = FEED_OUTPUT_DIR,
    audio_output_dir: Path = EPISODE_OUTPUT_DIR,
    overwrite: bool = False,
    list_only: bool = False,
) -> RssDownloadResult:
    feed = fetch_rss_feed(feed_url)
    podcast_dir = _prepare_podcast_directory(output_dir, feed)
    manifest_path = podcast_dir / MANIFEST_FILE_NAME
    initial_status = "not_requested" if list_only else "pending"
    entries = [
        EpisodeDownloadEntry(
            episode=episode,
            local_path=_build_audio_path(audio_output_dir, feed.title, episode),
            status=initial_status,
        )
        for episode in feed.episodes
    ]
    _save_manifest(manifest_path, feed, entries, list_only=list_only)

    if not list_only:
        _download_episodes(
            feed,
            manifest_path,
            entries,
            audio_output_dir=audio_output_dir,
            overwrite=overwrite,
        )
    return _build_download_result(feed, manifest_path, entries)


def _download_episodes(
    feed: RssFeed,
    manifest_path: Path,
    entries: list[EpisodeDownloadEntry],
    *,
    audio_output_dir: Path,
    overwrite: bool,
) -> None:
    total_episodes = len(entries)
    for episode_number, entry in enumerate(entries, start=1):
        episode = entry.episode
        target_path = entry.local_path
        logger.info(
            "开始处理 RSS 节目音频：podcast=%s episode_progress=%d/%d "
            "episode_id=%s title=%s expected_bytes=%s path=%s",
            feed.title,
            episode_number,
            total_episodes,
            episode.episode_id,
            episode.title,
            episode.audio_length_bytes
            if episode.audio_length_bytes is not None
            else "unknown",
            target_path,
        )
        try:
            prepare_episode_audio_directory(
                audio_output_dir,
                episode.episode_id,
                podcast_title=feed.title,
                episode_title=episode.title,
            )
            skipped = store_episode_audio(
                episode.audio_url,
                target_path,
                overwrite=overwrite,
            )
        except EpisodeAudioDownloadError as error:
            entry.status = "failed"
            entry.error = str(error)
            logger.error(
                "下载 RSS 节目音频失败：podcast=%s episode_progress=%d/%d "
                "episode_id=%s title=%s error=%s",
                feed.title,
                episode_number,
                total_episodes,
                episode.episode_id,
                episode.title,
                error,
            )
        else:
            entry.status = "skipped" if skipped else "downloaded"
            logger.info(
                "RSS 节目音频处理完成：podcast=%s episode_progress=%d/%d "
                "episode_id=%s status=%s path=%s",
                feed.title,
                episode_number,
                total_episodes,
                episode.episode_id,
                entry.status,
                target_path,
            )
        _save_manifest(manifest_path, feed, entries, list_only=False)


def _build_audio_path(
    audio_output_dir: Path,
    podcast_title: str,
    episode: RssEpisode,
) -> Path:
    return (
        find_episode_output_dir(
            audio_output_dir,
            episode.episode_id,
            podcast_title=podcast_title,
            episode_title=episode.title,
        )
        / EPISODE_AUDIO_DIR_NAME
        / f"{SOURCE_FILE_STEM}{episode.extension}"
    )


def _save_manifest(
    path: Path,
    feed: RssFeed,
    entries: list[EpisodeDownloadEntry],
    *,
    list_only: bool,
) -> None:
    statuses = [entry.status for entry in entries]
    payload = {
        "feed_url": feed.source_url,
        "podcast_title": feed.title,
        "episode_count": len(feed.episodes),
        "declared_total_bytes": sum(
            episode.audio_length_bytes or 0 for episode in feed.episodes
        ),
        "list_only": list_only,
        "downloaded_count": statuses.count("downloaded"),
        "skipped_count": statuses.count("skipped"),
        "failed_count": statuses.count("failed"),
        "pending_count": statuses.count("pending"),
        "episodes": [entry.to_dict() for entry in entries],
    }
    atomic_write_json(path, payload, mode=DEFAULT_FILE_CREATION_MODE)


def _build_download_result(
    feed: RssFeed,
    manifest_path: Path,
    entries: list[EpisodeDownloadEntry],
) -> RssDownloadResult:
    statuses = [entry.status for entry in entries]
    return RssDownloadResult(
        feed_url=feed.source_url,
        podcast_title=feed.title,
        manifest_path=manifest_path,
        episode_count=len(feed.episodes),
        downloaded_count=statuses.count("downloaded"),
        skipped_count=statuses.count("skipped"),
        failed_count=statuses.count("failed"),
    )


def _prepare_podcast_directory(output_dir: Path, feed: RssFeed) -> Path:
    directory_name = _build_podcast_directory_name(feed)
    try:
        resolved_output_dir = output_dir.resolve()
        podcast_dir = resolved_output_dir / directory_name
        if podcast_dir.is_symlink():
            raise RssFeedError(PODCAST_DIR_SYMLINK_ERROR.format(path=podcast_dir))
        podcast_dir.mkdir(parents=True, exist_ok=True)
        actual_podcast_dir = podcast_dir.resolve()
        if actual_podcast_dir != podcast_dir:
            raise RssFeedError(
                PODCAST_DIR_REDIRECT_ERROR_TEMPLATE.format(
                    expected_path=podcast_dir,
                    actual_path=actual_podcast_dir,
                )
            )
        return podcast_dir
    except OSError as error:
        raise RssFeedError(
            PREPARE_PODCAST_DIR_ERROR_TEMPLATE.format(
                path=output_dir,
                error_message=str(error),
            )
        ) from error


def _build_podcast_directory_name(feed: RssFeed) -> str:
    feed_hash = hashlib.sha256(feed.source_url.encode()).hexdigest()[:FEED_HASH_LENGTH]
    return build_named_directory_name(feed.title, identifier=feed_hash)
