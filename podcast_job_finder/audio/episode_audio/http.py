from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final, TypeGuard

import requests

from podcast_job_finder.audio.episode_audio.errors import (
    EpisodeAudioDownloadError,
    EpisodeAudioNotFoundError,
)
from podcast_job_finder.environment import get_optional_env_value
from podcast_job_finder.errors import ConfigurationError
from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT


DOWNLOAD_CHUNK_SIZE_BYTES: Final = 1024 * 1024
DOWNLOAD_CONNECT_TIMEOUT_SECONDS: Final = 10
DOWNLOAD_READ_TIMEOUT_SECONDS: Final = 60
DOWNLOAD_MAX_ATTEMPTS: Final = 3
DOWNLOAD_RETRY_BASE_DELAY_SECONDS: Final = 2.0
DOWNLOAD_RETRY_MAX_DELAY_SECONDS: Final = 4.0
RETRYABLE_OS_CONNECTION_ERRORS: Final = (
    ConnectionResetError,
    ConnectionAbortedError,
    ConnectionRefusedError,
    BrokenPipeError,
    TimeoutError,
)
DEFAULT_DOWNLOAD_CONNECTIONS: Final = 4
MAX_DOWNLOAD_CONNECTIONS: Final = 16
DOWNLOAD_CONNECTIONS_ENV: Final = "EPISODE_AUDIO_DOWNLOAD_CONNECTIONS"
RANGE_HEADER_NAME: Final = "Range"
FALLBACK_RANGE_STATUS_CODES: Final = frozenset({200, 400, 403, 416})
NOT_FOUND_STATUS_CODE: Final = 404
RETRYABLE_TOO_MANY_REQUESTS_STATUS: Final = 429
RETRYABLE_SERVER_ERROR_STATUS_MIN: Final = 500
USER_AGENT_HEADER_NAME: Final = "User-Agent"
INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE: Final = (
    "下载响应的 Content-Type 不是音频：{url}，Content-Type={content_type}"
)
INVALID_AUDIO_PAYLOAD_ERROR_TEMPLATE: Final = "下载响应正文疑似{detected_type}：{url}"
REQUEST_AUDIO_ERROR_TEMPLATE: Final = "请求节目音频失败：{url}，{error_message}"
WRITE_AUDIO_ERROR_TEMPLATE: Final = "写入节目音频临时文件失败：{path}，{error_message}"
INVALID_DOWNLOAD_CONNECTIONS_ENV_TEMPLATE: Final = (
    "环境变量 {env_name} 必须是 1 到 {max_connections} 的整数。"
)
INVALID_DOWNLOAD_CONNECTIONS_VALUE_TEMPLATE: Final = (
    "download_connections 必须是 1 到 {max_connections} 的整数。"
)
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

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _DownloadProgressState:
    lock: threading.Lock
    downloaded_bytes: int = 0
    last_report_time: float = 0.0
    last_report_bytes: int = 0


@dataclass(slots=True)
class _AudioDownloadProgressLogger:
    logger: logging.Logger = field(repr=False)
    partial_path: Path
    content_length: str | None
    report_interval_seconds: float = 60
    _total_bytes: int | None = field(init=False, repr=False)
    _state: _DownloadProgressState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._total_bytes = self._parse_total_bytes()
        self._state = _DownloadProgressState(
            lock=threading.Lock(),
            last_report_time=time.monotonic(),
        )

    def update(self, downloaded_bytes: int) -> None:
        with self._state.lock:
            self._update_locked(downloaded_bytes)

    def add(self, byte_count: int) -> None:
        with self._state.lock:
            self._state.downloaded_bytes += byte_count
            self._update_locked(self._state.downloaded_bytes)

    def _update_locked(self, downloaded_bytes: int) -> None:
        current_time = time.monotonic()
        elapsed_seconds = current_time - self._state.last_report_time
        if elapsed_seconds < self.report_interval_seconds:
            return

        speed_mib_per_second = (
            (downloaded_bytes - self._state.last_report_bytes)
            / elapsed_seconds
            / (1024 * 1024)
        )
        self._log_progress(downloaded_bytes, speed_mib_per_second)
        self._state.last_report_time = current_time
        self._state.last_report_bytes = downloaded_bytes

    def _parse_total_bytes(self) -> int | None:
        try:
            total_bytes = int((self.content_length or "").strip())
        except ValueError:
            return None
        return total_bytes if total_bytes > 0 else None

    def _log_progress(
        self,
        downloaded_bytes: int,
        speed_mib_per_second: float,
    ) -> None:
        episode_id = self.partial_path.parent.name
        if self._total_bytes is None:
            self.logger.info(
                "节目音频下载进度：episode_id=%s downloaded_bytes=%d "
                "total_bytes=unknown speed_mib_per_second=%.2f partial_path=%s",
                episode_id,
                downloaded_bytes,
                speed_mib_per_second,
                self.partial_path,
            )
            return

        self.logger.info(
            "节目音频下载进度：episode_id=%s downloaded_bytes=%d total_bytes=%d "
            "progress=%.1f%% speed_mib_per_second=%.2f partial_path=%s",
            episode_id,
            downloaded_bytes,
            self._total_bytes,
            downloaded_bytes / self._total_bytes * 100,
            speed_mib_per_second,
            self.partial_path,
        )


def download_audio_content(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
    *,
    download_connections: int | None = None,
) -> int:
    connections = _resolve_download_connections(download_connections)
    try:
        return _download_audio_content_with_retry(
            source_url,
            partial_path,
            partial_file,
            connections,
        )
    except requests.RequestException as error:
        error_type = (
            EpisodeAudioNotFoundError
            if _is_not_found_request_error(error)
            else EpisodeAudioDownloadError
        )
        raise error_type(
            REQUEST_AUDIO_ERROR_TEMPLATE.format(
                url=source_url,
                error_message=str(error),
            )
        ) from error
    except BaseExceptionGroup as error:
        error_type = (
            EpisodeAudioNotFoundError
            if _is_not_found_error_group(error)
            else EpisodeAudioDownloadError
        )
        raise error_type(
            REQUEST_AUDIO_ERROR_TEMPLATE.format(
                url=source_url,
                error_message=_format_download_error_message(error),
            )
        ) from error
    except OSError as error:
        if isinstance(error, RETRYABLE_OS_CONNECTION_ERRORS):
            raise EpisodeAudioDownloadError(
                REQUEST_AUDIO_ERROR_TEMPLATE.format(
                    url=source_url,
                    error_message=str(error),
                )
            ) from error
        raise EpisodeAudioDownloadError(
            WRITE_AUDIO_ERROR_TEMPLATE.format(
                path=partial_path,
                error_message=str(error),
            )
        ) from error


def _download_audio_content_with_retry(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
    connections: int,
) -> int:
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            _reset_partial_file(partial_file)
            return _download_audio_content_once(
                source_url,
                partial_path,
                partial_file,
                connections,
            )
        except (OSError, BaseExceptionGroup) as error:
            if attempt == DOWNLOAD_MAX_ATTEMPTS or not _is_retryable_download_error(
                error
            ):
                raise
            _wait_before_download_retry(source_url, attempt, error)
    raise AssertionError("节目音频下载重试循环未返回结果。")


def _reset_partial_file(partial_file: BinaryIO) -> None:
    partial_file.seek(0)
    partial_file.truncate(0)


class _SequentialDownloadRequired(Exception):
    """Raised when a range request cannot be used and the whole file must be fetched."""


def _resolve_download_connections(download_connections: int | None) -> int:
    if download_connections is None:
        return _load_download_connections_from_env()
    _ensure_valid_download_connections(
        download_connections,
        INVALID_DOWNLOAD_CONNECTIONS_VALUE_TEMPLATE.format(
            max_connections=MAX_DOWNLOAD_CONNECTIONS,
        ),
    )
    return download_connections


def _load_download_connections_from_env() -> int:
    try:
        parsed_connections = get_optional_env_value(DOWNLOAD_CONNECTIONS_ENV, int)
    except ValueError as error:
        raise _build_download_connections_env_error() from error
    if parsed_connections is None:
        return DEFAULT_DOWNLOAD_CONNECTIONS
    _ensure_valid_download_connections(
        parsed_connections,
        INVALID_DOWNLOAD_CONNECTIONS_ENV_TEMPLATE.format(
            env_name=DOWNLOAD_CONNECTIONS_ENV,
            max_connections=MAX_DOWNLOAD_CONNECTIONS,
        ),
    )
    return parsed_connections


def _ensure_valid_download_connections(
    download_connections: int,
    error_message: str,
) -> None:
    if download_connections < 1 or download_connections > MAX_DOWNLOAD_CONNECTIONS:
        raise ConfigurationError(error_message)


def _build_download_connections_env_error() -> ConfigurationError:
    return ConfigurationError(
        INVALID_DOWNLOAD_CONNECTIONS_ENV_TEMPLATE.format(
            env_name=DOWNLOAD_CONNECTIONS_ENV,
            max_connections=MAX_DOWNLOAD_CONNECTIONS,
        )
    )


def _download_audio_content_once(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
    connections: int,
) -> int:
    if connections <= 1:
        return _write_response_to_file(source_url, partial_path, partial_file)
    return _download_with_optional_ranges(
        source_url,
        partial_path,
        partial_file,
        connections,
    )


def _is_retryable_download_error(error: BaseException) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return bool(error.exceptions) and all(
            _is_retryable_download_error(inner) for inner in error.exceptions
        )
    if isinstance(error, requests.RequestException):
        return _is_retryable_request_error(error)
    return isinstance(error, RETRYABLE_OS_CONNECTION_ERRORS)


def _is_retryable_request_error(error: requests.RequestException) -> bool:
    if isinstance(
        error,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    if not isinstance(error, requests.HTTPError) or error.response is None:
        return False
    status_code = error.response.status_code
    return (
        status_code in (NOT_FOUND_STATUS_CODE, RETRYABLE_TOO_MANY_REQUESTS_STATUS)
        or status_code >= RETRYABLE_SERVER_ERROR_STATUS_MIN
    )


def _is_not_found_request_error(error: requests.RequestException) -> bool:
    return (
        isinstance(error, requests.HTTPError)
        and error.response is not None
        and error.response.status_code == NOT_FOUND_STATUS_CODE
    )


def _is_not_found_error_group(error: BaseExceptionGroup) -> bool:
    return bool(error.exceptions) and all(
        (
            _is_not_found_error_group(inner)
            if isinstance(inner, BaseExceptionGroup)
            else isinstance(inner, requests.RequestException)
            and _is_not_found_request_error(inner)
        )
        for inner in error.exceptions
    )


def _wait_before_download_retry(
    source_url: str,
    attempt: int,
    error: BaseException,
) -> None:
    delay_seconds = min(
        DOWNLOAD_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        DOWNLOAD_RETRY_MAX_DELAY_SECONDS,
    )
    logger.warning(
        "节目音频下载第 %d 次尝试失败，将在 %.2f 秒后重试：url=%s error=%s",
        attempt,
        delay_seconds,
        source_url,
        _format_download_error_message(error),
    )
    time.sleep(delay_seconds)


def _format_download_error_message(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup) and error.exceptions:
        return _format_download_error_message(error.exceptions[0])
    return str(error)


def _download_with_optional_ranges(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
    connections: int,
) -> int:
    with _open_audio_response(source_url, byte_range="0-0") as response:
        response.raise_for_status()
        _validate_response_content_type(response, source_url)
        if response.status_code != 206:
            logger.info(
                "服务器未按分段返回，改用单连接下载：url=%s status_code=%s",
                source_url,
                response.status_code,
            )
            return _write_streamed_response(
                response,
                source_url,
                partial_path,
                partial_file,
            )
        total_bytes = _parse_content_range_total(response.headers.get("Content-Range"))
        ranges: tuple[tuple[int, int], ...] | None = None
        parallel_total_bytes = 0
        if not _should_use_parallel_download(total_bytes, connections):
            logger.info(
                "节目音频不满足分段下载条件，改用单连接下载：url=%s total_bytes=%s connections=%d",
                source_url,
                total_bytes if total_bytes is not None else "unknown",
                connections,
            )
            ranges = None
        else:
            parallel_total_bytes = total_bytes
            ranges = _build_byte_ranges(parallel_total_bytes, connections)
    if ranges is None:
        return _write_response_to_file(source_url, partial_path, partial_file)
    try:
        return _download_ranges_parallel(
            source_url,
            partial_path,
            partial_file,
            ranges,
            parallel_total_bytes,
        )
    except _SequentialDownloadRequired:
        logger.info("分段下载未被服务器接受，改用单连接下载：url=%s", source_url)
        _reset_partial_file(partial_file)
        return _write_response_to_file(source_url, partial_path, partial_file)


def _should_use_parallel_download(
    total_bytes: int | None,
    connections: int,
) -> TypeGuard[int]:
    return (
        total_bytes is not None
        and total_bytes >= connections * DOWNLOAD_CHUNK_SIZE_BYTES
    )


def _parse_content_range_total(content_range: str | None) -> int | None:
    if content_range is None:
        return None
    normalized = content_range.strip()
    if not normalized.lower().startswith("bytes "):
        return None
    _, _, spec = normalized.partition(" ")
    _, separator, total = spec.partition("/")
    if separator != "/" or total in {"", "*"}:
        return None
    try:
        total_bytes = int(total)
    except ValueError:
        return None
    return total_bytes if total_bytes > 0 else None


def _build_byte_ranges(
    total_bytes: int, connections: int
) -> tuple[tuple[int, int], ...]:
    part_count = min(connections, total_bytes)
    base_size, remainder = divmod(total_bytes, part_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(part_count):
        size = base_size + (1 if index < remainder else 0)
        end = start + size - 1
        ranges.append((start, end))
        start = end + 1
    return tuple(ranges)


def _download_ranges_parallel(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
    ranges: tuple[tuple[int, int], ...],
    total_bytes: int,
) -> int:
    logger.info(
        "节目音频分段下载：connections=%d total_bytes=%d partial_path=%s",
        len(ranges),
        total_bytes,
        partial_path,
    )
    progress_logger = _AudioDownloadProgressLogger(
        logger=logger,
        partial_path=partial_path,
        content_length=str(total_bytes),
    )
    partial_file.truncate(total_bytes)
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [
            executor.submit(
                _download_byte_range,
                source_url,
                partial_path,
                start,
                end,
                progress_logger,
            )
            for start, end in ranges
        ]
        _wait_for_range_futures(futures)
    return total_bytes


def _wait_for_range_futures(futures: list[Future[None]]) -> None:
    fallback = False
    errors: list[Exception] = []
    for future in as_completed(futures):
        try:
            future.result()
        except _SequentialDownloadRequired:
            fallback = True
        except Exception as error:  # pylint: disable=broad-exception-caught
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise ExceptionGroup("节目音频分段下载失败", errors)
    if fallback:
        raise _SequentialDownloadRequired


def _download_byte_range(
    source_url: str,
    partial_path: Path,
    start: int,
    end: int,
    progress_logger: _AudioDownloadProgressLogger,
) -> None:
    expected_bytes = end - start + 1
    downloaded_bytes = 0
    with _open_audio_response(source_url, byte_range=f"{start}-{end}") as response:
        _raise_if_range_unsupported(response)
        _validate_response_content_type(response, source_url)
        with partial_path.open("rb+") as range_file:
            range_file.seek(start)
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE_BYTES):
                if not chunk:
                    continue
                if start == 0 and downloaded_bytes == 0:
                    _validate_first_chunk(chunk, source_url)
                range_file.write(chunk)
                downloaded_bytes += len(chunk)
                progress_logger.add(len(chunk))
    if downloaded_bytes != expected_bytes:
        raise requests.ConnectionError(
            "分段下载字节数不完整：url="
            f"{source_url} range={start}-{end} expected_bytes={expected_bytes} "
            f"actual_bytes={downloaded_bytes}"
        )


def _raise_if_range_unsupported(response: requests.Response) -> None:
    if response.status_code in FALLBACK_RANGE_STATUS_CODES:
        raise _SequentialDownloadRequired
    response.raise_for_status()
    if response.status_code != 206:
        raise _SequentialDownloadRequired


def _open_audio_response(
    source_url: str,
    *,
    byte_range: str | None = None,
) -> requests.Response:
    headers = {USER_AGENT_HEADER_NAME: DEFAULT_BROWSER_USER_AGENT}
    if byte_range is not None:
        headers[RANGE_HEADER_NAME] = f"bytes={byte_range}"
    return requests.get(
        source_url,
        headers=headers,
        stream=True,
        timeout=(DOWNLOAD_CONNECT_TIMEOUT_SECONDS, DOWNLOAD_READ_TIMEOUT_SECONDS),
    )


def _write_response_to_file(
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
) -> int:
    with _open_audio_response(source_url) as response:
        response.raise_for_status()
        return _write_streamed_response(
            response,
            source_url,
            partial_path,
            partial_file,
        )


def _write_streamed_response(
    response: requests.Response,
    source_url: str,
    partial_path: Path,
    partial_file: BinaryIO,
) -> int:
    _validate_response_content_type(response, source_url)
    progress_logger = _AudioDownloadProgressLogger(
        logger=logger,
        partial_path=partial_path,
        content_length=response.headers.get("Content-Length"),
    )
    downloaded_bytes = 0
    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE_BYTES):
        if not chunk:
            continue
        if downloaded_bytes == 0:
            _validate_first_chunk(chunk, source_url)
        partial_file.write(chunk)
        downloaded_bytes += len(chunk)
        progress_logger.update(downloaded_bytes)
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
