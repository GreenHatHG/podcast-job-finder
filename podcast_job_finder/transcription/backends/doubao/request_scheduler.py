from __future__ import annotations

import logging
import math
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Event, Thread
from time import monotonic
from typing import Generic, TypeVar


Request = TypeVar("Request")
Response = TypeVar("Response")
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ScheduledJob(Generic[Request, Response]):
    request: Request
    future: Future[Response]
    started: Event
    started_at: float | None = None


class DoubaoRequestScheduler(Generic[Request, Response]):  # pylint: disable=too-many-instance-attributes
    """按固定间隔启动豆包请求，并限制同时进行的请求数量。

    提交只把任务放进队列。单独的发送线程等到间隔到达、并且同时进行的数量
    未满时，才把任务交给工作线程。工作线程只负责调用豆包，不再等待间隔。
    """

    def __init__(
        self,
        *,
        max_in_flight_requests: int,
        interval_seconds: float,
        worker: Callable[[Request], Response],
    ) -> None:
        if max_in_flight_requests <= 0:
            raise ValueError("max_in_flight_requests 必须大于 0。")
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds 必须是大于 0 的有限数值。")
        self._interval_seconds = interval_seconds
        self._max_in_flight_requests = max_in_flight_requests
        self._worker = worker
        self._high_priority_jobs: deque[_ScheduledJob[Request, Response]] = deque()
        self._normal_jobs: deque[_ScheduledJob[Request, Response]] = deque()
        self._in_flight_requests = 0
        self._next_allowed_at: float | None = None
        self._closed = False
        self._condition = Condition()
        self._executor = ThreadPoolExecutor(max_workers=max_in_flight_requests)
        self._dispatcher = Thread(
            target=self._dispatch_loop,
            name="doubao-request-dispatcher",
        )
        self._dispatcher.start()

    def submit(
        self,
        request: Request,
        *,
        high_priority: bool = False,
    ) -> Future[Response]:
        future: Future[Response] = Future()
        job = _ScheduledJob(request, future, Event())
        with self._condition:
            if self._closed:
                raise RuntimeError("豆包请求调度器已经关闭。")
            if high_priority:
                self._high_priority_jobs.append(job)
            else:
                self._normal_jobs.append(job)
            # 发送线程可能正在等队列里出现新任务，这里唤醒它重新检查。
            self._condition.notify()
        return future

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending_jobs = self._take_pending_jobs()
            self._condition.notify_all()
        _cancel_jobs(pending_jobs)
        self._dispatcher.join()
        self._executor.shutdown(wait=True)

    def cancel_pending(self) -> None:
        """取消尚未交给工作线程的请求。已经开始的请求继续运行。"""

        with self._condition:
            pending_jobs = self._take_pending_jobs()
            self._condition.notify_all()
        _cancel_jobs(pending_jobs)

    def _dispatch_loop(self) -> None:
        try:
            while self._dispatch_next_job():
                pass
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("豆包请求发送循环异常退出")
            self._abort_pending_jobs(error)

    def _dispatch_next_job(self) -> bool:
        with self._condition:
            job = self._wait_for_dispatch_job()
            if job is None:
                return False
            self._in_flight_requests += 1
        self._start_job(job)
        job.started.wait()
        if job.started_at is None:
            raise AssertionError("豆包请求工作线程启动后没有记录开始时间。")
        with self._condition:
            self._next_allowed_at = job.started_at + self._interval_seconds
        return True

    def _wait_for_dispatch_job(self) -> _ScheduledJob[Request, Response] | None:
        # 线程被唤醒不代表已经获得发送资格。其他线程可能先一步修改状态，
        # 所以每次 wait() 返回后都要从头检查关闭、同时进行数量和发送时间。
        while True:
            if self._closed:
                return None
            if self._in_flight_requests >= self._max_in_flight_requests:
                # 等待期间会释放 Condition 内部的锁，让已经完成请求的线程能够
                # 归还名额。重新取得锁后，wait() 才会返回。
                self._condition.wait()
                continue

            if not self._high_priority_jobs and not self._normal_jobs:
                self._condition.wait()
                continue

            if self._next_allowed_at is not None:
                delay = self._next_allowed_at - monotonic()
                if delay > 0:
                    # 新任务或请求结束可能提前唤醒线程；醒来后仍要重新检查时间。
                    self._condition.wait(delay)
                    continue

            jobs = (
                self._high_priority_jobs
                if self._high_priority_jobs
                else self._normal_jobs
            )
            job = jobs.popleft()
            if not job.future.cancelled():
                return job

    def _start_job(self, job: _ScheduledJob[Request, Response]) -> None:
        try:
            self._executor.submit(self._run_job, job)
        except RuntimeError as error:
            if not job.future.done():
                job.future.set_exception(error)
            self._release_in_flight()
            raise

    def _run_job(self, job: _ScheduledJob[Request, Response]) -> None:
        job.started_at = monotonic()
        job.started.set()
        try:
            job.future.set_result(self._worker(job.request))
        except Exception as error:  # pylint: disable=broad-exception-caught
            if not job.future.done():
                job.future.set_exception(error)
        finally:
            self._release_in_flight()

    def _release_in_flight(self) -> None:
        with self._condition:
            self._in_flight_requests -= 1
            self._condition.notify_all()

    def _take_pending_jobs(self) -> list[_ScheduledJob[Request, Response]]:
        pending_jobs = [*self._high_priority_jobs, *self._normal_jobs]
        self._high_priority_jobs.clear()
        self._normal_jobs.clear()
        return pending_jobs

    def _abort_pending_jobs(self, error: Exception) -> None:
        with self._condition:
            self._closed = True
            pending_jobs = self._take_pending_jobs()
            self._condition.notify_all()
        for job in pending_jobs:
            if not job.future.done():
                job.future.set_exception(error)


def _cancel_jobs(jobs: list[_ScheduledJob[Request, Response]]) -> None:
    for job in jobs:
        job.future.cancel()
