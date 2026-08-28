from __future__ import annotations

# pylint: disable=protected-access

from collections.abc import Generator, Sequence
from concurrent.futures import Future
from pathlib import Path
from threading import Lock
from typing import cast
from unittest import TestCase
from unittest.mock import Mock

from podcast_job_finder.audio.segmentation.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.segmentation.vad import SpeechSegment
from podcast_job_finder.transcription.backends.doubao.transcriber import (
    DoubaoAudioTranscriber,
    DoubaoRequestError,
    DoubaoServiceErrorThresholdExceeded,
    _PendingRequest,
)
from podcast_job_finder.transcription.backends.doubao.request_client import (
    SessionResponses,
)
from podcast_job_finder.transcription.models import (
    TranscribedSpeechSegment,
    TranscriptionOutput,
    TranscriptionSegmentResult,
)


class DoubaoBatchFailureTest(TestCase):
    def test_failed_segment_and_next_result_are_not_saved(self) -> None:
        segments = [_segment(index) for index in range(1, 6)]
        futures = [_successful_future() for _ in segments]
        futures[1] = _failed_future(RuntimeError("segment 2 failed"))
        transcriber = object.__new__(DoubaoAudioTranscriber)
        transcriber._request_scheduler = Mock()  # type: ignore[attr-defined]
        transcriber._submit_segment = Mock(side_effect=futures)  # type: ignore[method-assign]
        transcriber._consume_pending_request = Mock(  # type: ignore[method-assign]
            side_effect=[
                _successful_result(segments[0]),
                (
                    None,
                    _segment_context(segments[1]),
                    RuntimeError("segment 2 failed"),
                ),
                _successful_result(segments[2]),
                _successful_result(segments[3]),
                _successful_result(segments[4]),
            ]
        )

        iterator = transcriber.transcribe_batches(segments, previous_segment=None)
        yielded: list[TranscriptionSegmentResult] = []
        with self.assertRaisesRegex(RuntimeError, "segment 2 failed"):
            while True:
                yielded.extend(next(iterator))

        self.assertEqual([item.segment.index for item in yielded], [1, 4, 5])
        self.assertEqual(transcriber._consume_pending_request.call_count, 5)
        transcriber._request_scheduler.cancel_pending.assert_called_once_with()

    def test_service_error_threshold_stops_batch_immediately(self) -> None:
        segments = [_segment(index) for index in range(1, 5)]
        futures = [_successful_future() for _ in segments]
        threshold_error = DoubaoServiceErrorThresholdExceeded("threshold reached")
        transcriber = object.__new__(DoubaoAudioTranscriber)
        transcriber._request_scheduler = Mock()  # type: ignore[attr-defined]
        transcriber._submit_segment = Mock(side_effect=futures)  # type: ignore[method-assign]
        transcriber._consume_pending_request = Mock(  # type: ignore[method-assign]
            side_effect=[
                _successful_result(segments[0]),
                (None, _segment_context(segments[1]), threshold_error),
                _successful_result(segments[2]),
                _successful_result(segments[3]),
            ]
        )

        iterator = transcriber.transcribe_batches(segments, previous_segment=None)
        self.assertEqual([item.segment.index for item in next(iterator)], [1])
        with self.assertRaisesRegex(
            DoubaoServiceErrorThresholdExceeded,
            "threshold reached",
        ):
            next(iterator)

        self.assertEqual(transcriber._consume_pending_request.call_count, 2)
        transcriber._request_scheduler.cancel_pending.assert_called_once_with()

    def test_unexpected_request_error_propagates_immediately(self) -> None:
        segment = _segment(1)
        transcriber = object.__new__(DoubaoAudioTranscriber)
        future = _failed_future(RuntimeError("unexpected bug"))

        with self.assertRaisesRegex(RuntimeError, "unexpected bug"):
            transcriber._consume_pending_request(
                _PendingRequest(segment, future),
                previous_segment=None,
            )

    def test_unexpected_processing_error_propagates_immediately(self) -> None:
        segment = _segment(1)
        transcriber = object.__new__(DoubaoAudioTranscriber)
        transcriber._build_transcription_item = Mock(  # type: ignore[method-assign]
            side_effect=RuntimeError("unexpected processing bug")
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected processing bug"):
            transcriber._consume_pending_request(
                _PendingRequest(segment, _successful_future()),
                previous_segment=None,
            )

    def test_unexpected_batch_error_cancels_pending_requests(self) -> None:
        segments = [_segment(index) for index in range(1, 4)]
        transcriber = object.__new__(DoubaoAudioTranscriber)
        transcriber._request_scheduler = Mock()  # type: ignore[attr-defined]
        transcriber._submit_segment = Mock(  # type: ignore[method-assign]
            side_effect=[_successful_future() for _ in segments]
        )
        transcriber._consume_pending_request = Mock(  # type: ignore[method-assign]
            side_effect=RuntimeError("unexpected batch bug")
        )

        with self.assertRaisesRegex(RuntimeError, "unexpected batch bug"):
            next(transcriber.transcribe_batches(segments, previous_segment=None))

        transcriber._request_scheduler.cancel_pending.assert_called_once_with()

    def test_closing_batch_iterator_cancels_pending_requests(self) -> None:
        segments = [_segment(index) for index in range(1, 4)]
        transcriber = object.__new__(DoubaoAudioTranscriber)
        transcriber._request_scheduler = Mock()  # type: ignore[attr-defined]
        transcriber._submit_segment = Mock(  # type: ignore[method-assign]
            side_effect=[_successful_future() for _ in segments]
        )
        transcriber._consume_pending_request = Mock(  # type: ignore[method-assign]
            side_effect=[_successful_result(segment) for segment in segments]
        )
        iterator = cast(
            Generator[Sequence[TranscriptionSegmentResult], None, None],
            transcriber.transcribe_batches(segments, previous_segment=None),
        )

        self.assertEqual([item.segment.index for item in next(iterator)], [1])
        iterator.close()

        transcriber._request_scheduler.cancel_pending.assert_called_once_with()

    def test_later_request_failures_reach_threshold_before_ordered_consumption(
        self,
    ) -> None:
        futures = [_successful_future()]
        futures.extend(
            _failed_future(DoubaoRequestError(f"failure {index}"))
            for index in range(2, 7)
        )
        futures.append(_successful_future())
        transcriber = object.__new__(DoubaoAudioTranscriber)
        transcriber._request_scheduler = Mock(  # type: ignore[attr-defined]
            submit=Mock(side_effect=futures)
        )
        transcriber._service_error_segment_paths = set()  # type: ignore[attr-defined]
        transcriber._service_error_lock = Lock()  # type: ignore[attr-defined]
        transcriber._service_error_threshold_error = None  # type: ignore[attr-defined]
        segments = [_segment(index) for index in range(1, 8)]

        with self.assertRaises(DoubaoServiceErrorThresholdExceeded):
            for segment in segments:
                transcriber._submit_segment(segment, total_segment_count=7)

        self.assertEqual(len(transcriber._service_error_segment_paths), 5)
        self.assertEqual(transcriber._request_scheduler.submit.call_count, 6)
        transcriber._request_scheduler.cancel_pending.assert_called_once_with()


def _segment(index: int) -> ExportedSpeechSegment:
    return ExportedSpeechSegment(
        index=index,
        segment=SpeechSegment(
            start_sample=(index - 1) * 16_000,
            end_sample=index * 16_000,
        ),
        file_path=Path(f"segment-{index}.wav"),
    )


def _successful_future() -> Future[SessionResponses]:
    future: Future[SessionResponses] = Future()
    future.set_result([])
    return future


def _failed_future(error: Exception) -> Future[SessionResponses]:
    future: Future[SessionResponses] = Future()
    future.set_exception(error)
    return future


def _successful_result(
    segment: ExportedSpeechSegment,
) -> tuple[TranscriptionSegmentResult, TranscribedSpeechSegment, None]:
    result = TranscriptionSegmentResult(
        segment=segment,
        output=TranscriptionOutput(text=f"segment {segment.index}"),
        previous_segment=None,
    )
    return result, _segment_context(segment), None


def _segment_context(segment: ExportedSpeechSegment) -> TranscribedSpeechSegment:
    return TranscribedSpeechSegment(
        index=segment.index,
        start_ms=segment.segment.start_ms,
        end_ms=segment.segment.end_ms,
        text="",
    )
