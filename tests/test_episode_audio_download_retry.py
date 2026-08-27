from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, patch

import requests

from podcast_job_finder.audio.episode_audio.errors import EpisodeAudioDownloadError
from podcast_job_finder.audio.episode_audio.http import (
    INVALID_AUDIO_CONTENT_TYPE_ERROR_TEMPLATE,
    REQUEST_AUDIO_ERROR_TEMPLATE,
    download_audio_content,
)


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
    ) -> None:
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(content)),
        }
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
                download_audio_content(AUDIO_URL, partial_path, partial_file)

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
                download_audio_content(AUDIO_URL, partial_path, partial_file)

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
                download_audio_content(AUDIO_URL, partial_path, partial_file)

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
            )
            partial_file.flush()
            partial_file.seek(0)
            return downloaded_bytes, partial_file.read()
