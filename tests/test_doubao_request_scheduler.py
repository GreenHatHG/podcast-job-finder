from __future__ import annotations

import time
from concurrent.futures import Future
from threading import Event, Lock
from threading import Thread
from time import monotonic
from unittest import TestCase
from unittest.mock import patch

from podcast_job_finder.transcription.backends.doubao import request_scheduler
from podcast_job_finder.transcription.backends.doubao.request_scheduler import (
    DoubaoRequestScheduler,
)


class DoubaoRequestSchedulerTest(TestCase):
    def test_interval_starts_when_worker_thread_really_starts(self) -> None:
        starts: list[float] = []

        def worker(request: int) -> int:
            starts.append(monotonic())
            return request

        with patch.object(
            request_scheduler,
            "ThreadPoolExecutor",
            _DelayedFirstStartExecutor,
        ):
            scheduler = DoubaoRequestScheduler(
                max_in_flight_requests=2,
                interval_seconds=0.05,
                worker=worker,
            )
            futures = [scheduler.submit(index) for index in range(2)]
            self.assertEqual(
                [future.result(timeout=2) for future in futures],
                [0, 1],
            )
            scheduler.close()

        self.assertEqual(len(starts), 2)
        self.assertGreaterEqual(starts[1] - starts[0], 0.04)

    def test_long_worker_does_not_delay_next_dispatch(self) -> None:
        starts: list[float] = []
        lock = Lock()

        def worker(request: int) -> int:
            with lock:
                starts.append(monotonic())
            if request == 0:
                time.sleep(0.2)
            return request

        scheduler = DoubaoRequestScheduler(
            max_in_flight_requests=3,
            interval_seconds=0.05,
            worker=worker,
        )
        self.addCleanup(scheduler.close)
        futures = [scheduler.submit(index) for index in range(3)]
        self.assertEqual([future.result(timeout=2) for future in futures], [0, 1, 2])
        self.assertEqual(len(starts), 3)
        first_gap = starts[1] - starts[0]
        second_gap = starts[2] - starts[1]
        self.assertGreaterEqual(first_gap, 0.04)
        self.assertLess(first_gap, 0.15)
        self.assertGreaterEqual(second_gap, 0.04)
        self.assertLess(second_gap, 0.15)

    def test_max_in_flight_requests_caps_running_workers(self) -> None:
        running = 0
        max_running = 0
        lock = Lock()
        release = Event()
        started_two = Event()

        def worker(request: int) -> int:
            nonlocal running, max_running
            with lock:
                running += 1
                max_running = max(max_running, running)
                if max_running >= 2:
                    started_two.set()
            self.assertTrue(release.wait(timeout=2))
            with lock:
                running -= 1
            return request

        scheduler = DoubaoRequestScheduler(
            max_in_flight_requests=2,
            interval_seconds=0.01,
            worker=worker,
        )
        self.addCleanup(scheduler.close)
        futures = [scheduler.submit(index) for index in range(4)]
        self.assertTrue(started_two.wait(timeout=2))
        time.sleep(0.05)
        with lock:
            self.assertEqual(max_running, 2)
            self.assertLessEqual(running, 2)
        release.set()
        self.assertEqual([future.result(timeout=2) for future in futures], [0, 1, 2, 3])

    def test_full_capacity_sends_immediately_then_repaces(self) -> None:
        starts: list[float] = []
        lock = Lock()

        def worker(request: int) -> int:
            with lock:
                starts.append(monotonic())
            if request == 0:
                time.sleep(0.2)
            return request

        scheduler = DoubaoRequestScheduler(
            max_in_flight_requests=1,
            interval_seconds=0.08,
            worker=worker,
        )
        self.addCleanup(scheduler.close)
        futures = [scheduler.submit(index) for index in range(3)]
        self.assertEqual([future.result(timeout=3) for future in futures], [0, 1, 2])
        self.assertEqual(len(starts), 3)
        self.assertGreaterEqual(starts[1] - starts[0], 0.18)
        self.assertGreaterEqual(starts[2] - starts[1], 0.06)
        self.assertLess(starts[2] - starts[1], 0.16)

    def test_cancel_pending_skips_unsent_jobs(self) -> None:
        started = Event()
        release = Event()

        def worker(request: str) -> str:
            if request == "first":
                started.set()
                self.assertTrue(release.wait(timeout=2))
            return request

        scheduler = DoubaoRequestScheduler(
            max_in_flight_requests=1,
            interval_seconds=0.05,
            worker=worker,
        )
        self.addCleanup(scheduler.close)
        first = scheduler.submit("first")
        second = scheduler.submit("second")
        third = scheduler.submit("third")
        self.assertTrue(started.wait(timeout=2))
        scheduler.cancel_pending()
        release.set()
        self.assertEqual(first.result(timeout=2), "first")
        self.assertTrue(second.cancelled())
        self.assertTrue(third.cancelled())

    def test_high_priority_job_is_dispatched_before_queued_normal_jobs(self) -> None:
        started = Event()
        release = Event()
        order: list[str] = []
        lock = Lock()

        def worker(request: str) -> str:
            if request == "first":
                started.set()
                self.assertTrue(release.wait(timeout=2))
            with lock:
                order.append(request)
            return request

        scheduler = DoubaoRequestScheduler(
            max_in_flight_requests=1,
            interval_seconds=0.05,
            worker=worker,
        )
        self.addCleanup(scheduler.close)
        first = scheduler.submit("first")
        second = scheduler.submit("second")
        third = scheduler.submit("third")
        self.assertTrue(started.wait(timeout=2))
        priority = scheduler.submit("priority", high_priority=True)
        release.set()
        self.assertEqual(first.result(timeout=2), "first")
        self.assertEqual(priority.result(timeout=2), "priority")
        self.assertEqual(second.result(timeout=2), "second")
        self.assertEqual(third.result(timeout=2), "third")
        self.assertEqual(order, ["first", "priority", "second", "third"])


class _DelayedFirstStartExecutor:
    def __init__(self, *, max_workers: int) -> None:
        del max_workers
        self._submit_count = 0
        self._threads: list[Thread] = []

    def submit(self, function: object, *args: object) -> Future[None]:
        self._submit_count += 1
        delay = 0.15 if self._submit_count == 1 else 0
        result: Future[None] = Future()

        def run() -> None:
            time.sleep(delay)
            try:
                function(*args)  # type: ignore[operator]
            except Exception as error:  # pylint: disable=broad-exception-caught
                result.set_exception(error)
            else:
                result.set_result(None)

        thread = Thread(target=run)
        self._threads.append(thread)
        thread.start()
        return result

    def shutdown(self, *, wait: bool) -> None:
        if wait:
            for thread in self._threads:
                thread.join()
