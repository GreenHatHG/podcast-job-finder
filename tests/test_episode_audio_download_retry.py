from __future__ import annotations

from collections.abc import Iterator
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, patch

import requests

from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioDownloadError
from podcast_job_finder.audio.episode_audio.http import (
    DOWNLOAD_CONNECTIONS_ENV,
    INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE,
    INVALID_DOWNLOAD_CONNECTIONS_ENV_TEMPLATE,
    MAX_DOWNLOAD_CONNECTIONS,
    REQUEST_AUDIO_ERROR_TEMPLATE,
    download_audio_content,
)
from podcast_job_finder.errors import ConfigurationError


AUDIO_URL = "https://example.com/episode.m4a"
SUCCESS_AUDIO = b"ID3successful-audio"
PARTIAL_AUDIO = b"ID3partial-audio"
GET_PATCH_TARGET = "podcast_job_finder.audio.episode_audio.http.requests.get"
SLEEP_PATCH_TARGET = "podcast_job_finder.audio.episode_audio.http.time.sleep"


class FakeAudioResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        content_type: str = "audio/mp4",
        iter_error: BaseException | None = None,
        content_range: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(content)),
        }
        if content_range is not None:
            self.headers["Content-Range"] = content_range
        self._content = content
        self._iter_error = iter_error

    def __enter__(self) -> FakeAudioResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        raise requests.HTTPError(
            f"{self.status_code} Error for url: {AUDIO_URL}",
            response=self,
        )

    def iter_content(self, chunk_size: int = 0) -> Iterator[bytes]:
        del chunk_size
        if self._content:
            yield self._content
        if self._iter_error is not None:
            raise self._iter_error


class EpisodeAudioDownloadRetryTest(TestCase):
    def test_retries_read_timeout_then_keeps_successful_bytes(self) -> None:
        timeout = requests.ReadTimeout("Read timed out.")
        with (
            patch(
                GET_PATCH_TARGET,
                side_effect=[
                    FakeAudioResponse(PARTIAL_AUDIO, iter_error=timeout),
                    FakeAudioResponse(SUCCESS_AUDIO),
                ],
            ) as mock_get,
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
        ):
            downloaded_bytes, saved_content = _download_audio()

        self.assertEqual(downloaded_bytes, len(SUCCESS_AUDIO))
        self.assertEqual(saved_content, SUCCESS_AUDIO)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(2.0)

    def test_raises_after_three_read_timeouts(self) -> None:
        timeout = requests.ReadTimeout("Read timed out.")
        with (
            patch(
                GET_PATCH_TARGET, side_effect=[timeout, timeout, timeout]
            ) as mock_get,
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
            TemporaryDirectory() as temp_dir,
        ):
            partial_path = Path(temp_dir) / "source.part"
            with (
                partial_path.open("wb+") as partial_file,
                self.assertRaises(EpisodeAudioDownloadError) as raised,
            ):
                download_audio_content(
                    AUDIO_URL,
                    partial_path,
                    partial_file,
                    download_connections=1,
                )

        self.assertEqual(
            str(raised.exception),
            REQUEST_AUDIO_ERROR_TEMPLATE.format(
                url=AUDIO_URL,
                error_message="Read timed out.",
            ),
        )
        self.assertEqual(mock_get.call_count, 3)
        self.assertEqual(mock_sleep.call_args_list, [call(2.0), call(4.0)])

    def test_retries_http_503_then_succeeds(self) -> None:
        with (
            patch(
                GET_PATCH_TARGET,
                side_effect=[
                    FakeAudioResponse(b"", status_code=503),
                    FakeAudioResponse(SUCCESS_AUDIO),
                ],
            ) as mock_get,
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
        ):
            downloaded_bytes, saved_content = _download_audio()

        self.assertEqual(downloaded_bytes, len(SUCCESS_AUDIO))
        self.assertEqual(saved_content, SUCCESS_AUDIO)
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_called_once_with(2.0)

    def test_does_not_retry_http_404(self) -> None:
        with (
            patch(
                GET_PATCH_TARGET,
                return_value=FakeAudioResponse(b"", status_code=404),
            ) as mock_get,
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
            TemporaryDirectory() as temp_dir,
        ):
            partial_path = Path(temp_dir) / "source.part"
            with (
                partial_path.open("wb+") as partial_file,
                self.assertRaises(EpisodeAudioDownloadError) as raised,
            ):
                download_audio_content(
                    AUDIO_URL,
                    partial_path,
                    partial_file,
                    download_connections=1,
                )

        self.assertEqual(
            str(raised.exception),
            REQUEST_AUDIO_ERROR_TEMPLATE.format(
                url=AUDIO_URL,
                error_message="404 Error for url: https://example.com/episode.m4a",
            ),
        )
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()

    def test_does_not_retry_invalid_content_type(self) -> None:
        with (
            patch(
                GET_PATCH_TARGET,
                return_value=FakeAudioResponse(
                    b"<html>error</html>",
                    content_type="text/html",
                ),
            ) as mock_get,
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
            TemporaryDirectory() as temp_dir,
        ):
            partial_path = Path(temp_dir) / "source.part"
            with (
                partial_path.open("wb+") as partial_file,
                self.assertRaises(EpisodeAudioDownloadError) as raised,
            ):
                download_audio_content(
                    AUDIO_URL,
                    partial_path,
                    partial_file,
                    download_connections=1,
                )

        self.assertEqual(
            str(raised.exception),
            INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE.format(
                url=AUDIO_URL,
                content_type="text/html",
            ),
        )
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()


def _download_audio() -> tuple[int, bytes]:
    with TemporaryDirectory() as temp_dir:
        partial_path = Path(temp_dir) / "source.part"
        with partial_path.open("wb+") as partial_file:
            downloaded_bytes = download_audio_content(
                AUDIO_URL,
                partial_path,
                partial_file,
                download_connections=1,
            )
        return downloaded_bytes, partial_path.read_bytes()


PARALLEL_AUDIO = b"ABCDEFGH"
CHUNK_SIZE_PATCH_TARGET = (
    "podcast_job_finder.audio.episode_audio.http.DOWNLOAD_CHUNK_SIZE_BYTES"
)


class RangeAwareAudioGet:
    def __init__(
        self,
        content: bytes,
        *,
        timeout_range: str | None = None,
        range_status_code: int = 206,
    ) -> None:
        self._content = content
        self._timeout_range = timeout_range
        self._timeout_remaining = 1 if timeout_range is not None else 0
        self._range_status_code = range_status_code
        self._lock = threading.Lock()

    def __call__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        **kwargs: object,
    ) -> FakeAudioResponse:
        del url, kwargs
        range_header = (headers or {}).get("Range")
        if range_header == "bytes=0-0":
            return FakeAudioResponse(
                self._content[:1],
                status_code=206,
                content_range=f"bytes 0-0/{len(self._content)}",
            )
        if range_header is None:
            return FakeAudioResponse(self._content)
        start, end = _parse_range_header(range_header)
        with self._lock:
            if range_header == self._timeout_range and self._timeout_remaining > 0:
                self._timeout_remaining -= 1
                raise requests.ReadTimeout("Read timed out.")
        if self._range_status_code != 206:
            return FakeAudioResponse(b"", status_code=self._range_status_code)
        return FakeAudioResponse(
            self._content[start : end + 1],
            status_code=206,
            content_range=f"bytes {start}-{end}/{len(self._content)}",
        )


def _parse_range_header(range_header: str) -> tuple[int, int]:
    spec = range_header.removeprefix("bytes=")
    start_text, _, end_text = spec.partition("-")
    return int(start_text), int(end_text)


class EpisodeAudioRangedDownloadTest(TestCase):
    def test_assembles_range_responses_in_byte_order(self) -> None:
        with (
            patch(GET_PATCH_TARGET, side_effect=RangeAwareAudioGet(PARALLEL_AUDIO)),
            patch(CHUNK_SIZE_PATCH_TARGET, 2),
        ):
            downloaded_bytes, saved_content = _download_ranged_audio()

        self.assertEqual(downloaded_bytes, len(PARALLEL_AUDIO))
        self.assertEqual(saved_content, PARALLEL_AUDIO)

    def test_uses_probe_body_when_server_ignores_range(self) -> None:
        with patch(
            GET_PATCH_TARGET,
            return_value=FakeAudioResponse(SUCCESS_AUDIO),
        ) as mock_get:
            downloaded_bytes, saved_content = _download_ranged_audio()

        self.assertEqual(downloaded_bytes, len(SUCCESS_AUDIO))
        self.assertEqual(saved_content, SUCCESS_AUDIO)
        self.assertEqual(mock_get.call_count, 1)

    def test_retries_after_range_read_timeout(self) -> None:
        getter = RangeAwareAudioGet(PARALLEL_AUDIO, timeout_range="bytes=0-1")
        with (
            patch(GET_PATCH_TARGET, side_effect=getter),
            patch(CHUNK_SIZE_PATCH_TARGET, 2),
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
        ):
            downloaded_bytes, saved_content = _download_ranged_audio()

        self.assertEqual(downloaded_bytes, len(PARALLEL_AUDIO))
        self.assertEqual(saved_content, PARALLEL_AUDIO)
        mock_sleep.assert_called_once_with(2.0)

    def test_falls_back_to_single_connection_when_range_returns_403(self) -> None:
        getter = RangeAwareAudioGet(PARALLEL_AUDIO, range_status_code=403)
        with (
            patch(GET_PATCH_TARGET, side_effect=getter),
            patch(CHUNK_SIZE_PATCH_TARGET, 2),
        ):
            downloaded_bytes, saved_content = _download_ranged_audio()

        self.assertEqual(downloaded_bytes, len(PARALLEL_AUDIO))
        self.assertEqual(saved_content, PARALLEL_AUDIO)

    def test_rejects_invalid_content_type_on_probe_without_retry(self) -> None:
        with (
            patch(
                GET_PATCH_TARGET,
                return_value=FakeAudioResponse(
                    b"<html>error</html>",
                    content_type="text/html",
                ),
            ) as mock_get,
            patch(SLEEP_PATCH_TARGET) as mock_sleep,
            TemporaryDirectory() as temp_dir,
        ):
            partial_path = Path(temp_dir) / "source.part"
            with (
                partial_path.open("wb+") as partial_file,
                self.assertRaises(EpisodeAudioDownloadError) as raised,
            ):
                download_audio_content(
                    AUDIO_URL,
                    partial_path,
                    partial_file,
                    download_connections=4,
                )

        self.assertEqual(
            str(raised.exception),
            INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE.format(
                url=AUDIO_URL,
                content_type="text/html",
            ),
        )
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()

    def test_rejects_invalid_download_connections_env(self) -> None:
        with (
            patch.dict(os.environ, {DOWNLOAD_CONNECTIONS_ENV: "0"}),
            TemporaryDirectory() as temp_dir,
            self.assertRaises(ConfigurationError) as raised,
        ):
            partial_path = Path(temp_dir) / "source.part"
            with partial_path.open("wb+") as partial_file:
                download_audio_content(AUDIO_URL, partial_path, partial_file)

        self.assertEqual(
            str(raised.exception),
            INVALID_DOWNLOAD_CONNECTIONS_ENV_TEMPLATE.format(
                env_name=DOWNLOAD_CONNECTIONS_ENV,
                max_connections=MAX_DOWNLOAD_CONNECTIONS,
            ),
        )


def _download_ranged_audio() -> tuple[int, bytes]:
    with TemporaryDirectory() as temp_dir:
        partial_path = Path(temp_dir) / "source.part"
        with partial_path.open("wb+") as partial_file:
            downloaded_bytes = download_audio_content(
                AUDIO_URL,
                partial_path,
                partial_file,
                download_connections=4,
            )
        return downloaded_bytes, partial_path.read_bytes()
