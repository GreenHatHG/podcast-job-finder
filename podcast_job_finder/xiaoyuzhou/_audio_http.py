from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Final

import requests

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT
from podcast_job_finder.xiaoyuzhou._audio_errors import EpisodeAudioDownloadError


DOWNLOAD_CHUNK_SIZE_BYTES: Final = 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT_SECONDS: Final = 10
DOWNLOAD_READ_TIMEOUT_SECONDS: Final = 60
USER_AGENT_HEADER_NAME: Final = "User-Agent"
INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE: Final = (
    "下载响应的 Content-Type 不是音频：{url}，Content-Type={content_type}"
)
INVALID_AUDIO_PAYLOAD_ERROR_TEMPLATE: Final = "下载响应正文疑似{detected_type}：{url}"
REQUEST_AUDIO_ERROR_TEMPLATE: Final = "请求节目音频失败：{url}，{error_message}"
WRITE_AUDIO_ERROR_TEMPLATE: Final = "写入节目音频临时文件失败：{path}，{error_message}"
REJECTED_CONTENT_TYPES: Final = frozenset(
    {
        "application/json",
        "application/xml",
        "text/xml",
    }
)
ERROR_PAYLOAD_SIGNATURES: Final = (
    (b"<!doctype html", "HTML 文档"),
    (b"<html", "HTML 文档"),
    (b"<?xml", "XML 文档"),
    (b"{", "JSON 对象"),
    (b"[", "JSON 数组"),
)


def download_audio_content(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
) -> int:
    try:
        return _write_response_to_file(source_url, partial_file)
    except requests.RequestException as error:
        raise EpisodeAudioDownloadError(
            REQUEST_AUDIO_ERROR_TEMPLATE.format(
                url=source_url,
                error_message=str(error),
            )
        ) from error
    except OSError as error:
        raise EpisodeAudioDownloadError(
            WRITE_AUDIO_ERROR_TEMPLATE.format(
                path=partial_path,
                error_message=str(error),
            )
        ) from error


def _write_response_to_file(source_url: str, partial_file: BinaryIO) -> int:
    downloaded_bytes = 0
    with requests.get(
        source_url,
        headers={USER_AGENT_HEADER_NAME: DEFAULT_BROWSER_USER_AGENT},
        stream=True,
        timeout=(DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS),
    ) as response:
        response.raise_for_status()
        _validate_response_content_type(response, source_url)
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE_BYTES):
            if not chunk:
                continue
            if downloaded_bytes == 0:
                _validate_first_chunk(chunk, source_url)
            partial_file.write(chunk)
            downloaded_bytes += len(chunk)
    return downloaded_bytes


def _validate_response_content_type(
    response: requests.Response,
    source_url: str,
) -> None:
    content_type = response.headers.get("Content-Type", "")
    normalized_content_type = content_type.partition(";")[0].strip().lower()
    if normalized_content_type.startswith("text/"):
        raise EpisodeAudioDownloadError(
            INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE.format(
                url=source_url,
                content_type=content_type,
            )
        )
    if normalized_content_type in REJECTED_CONTENT_TYPES:
        raise EpisodeAudioDownloadError(
            INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE.format(
                url=source_url,
                content_type=content_type,
            )
        )


def _validate_first_chunk(chunk: bytes, source_url: str) -> None:
    normalized_prefix = chunk[:64].lstrip().lower()
    for signature, detected_type in ERROR_PAYLOAD_SIGNATURES:
        if normalized_prefix.startswith(signature):
            raise EpisodeAudioDownloadError(
                INVALID_AUDIO_PAYLOAD_ERROR_TEMPLATE.format(
                    url=source_url,
                    detected_type=detected_type,
                )
            )
