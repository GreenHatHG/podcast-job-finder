from __future__ import annotations

from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioDownloadError


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
INVALID_AUDIO_URL_ERROR: Final = "节目音频 URL 无效：{url}"
UNSUPPORTED_AUDIO_EXTENSION_ERROR: Final = "不支持的节目音频扩展名：{url}"


def extract_audio_extension(source_url: str) -> str:
    parsed_url = urlparse(source_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise EpisodeAudioDownloadError(INVALID_AUDIO_URL_ERROR.format(url=source_url))

    extension = Path(parsed_url.path).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        raise EpisodeAudioDownloadError(
            UNSUPPORTED_AUDIO_EXTENSION_ERROR.format(url=source_url)
        )
    return extension
