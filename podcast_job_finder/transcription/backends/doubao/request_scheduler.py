from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from time import monotonic, sleep
from typing import Generic, TypeVar


Request = TypeVar("Request")
Response = TypeVar("Response")


class DoubaoRequestScheduler(Generic[Request, Response]):
    """按固定间隔启动豆包请求，并限制实际工作的线程数量。"""

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
        self._worker = worker
        self._executor = ThreadPoolExecutor(max_workers=max_in_flight_requests)
        self._dispatch_lock = Lock()
        self._last_dispatch_at: float | None = None
        self._closed = False

    def submit(self, request: Request) -> Future[Response]:
        if self._closed:
            raise RuntimeError("豆包请求调度器已经关闭。")
        return self._executor.submit(self._run_request, request)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True)

    def _wait_for_dispatch_slot(self) -> None:
        with self._dispatch_lock:
            now = monotonic()
            if self._last_dispatch_at is not None:
                delay = self._last_dispatch_at + self._interval_seconds - now
                if delay > 0:
                    sleep(delay)
            self._last_dispatch_at = monotonic()

    def _run_request(self, request: Request) -> Response:
        self._wait_for_dispatch_slot()
        return self._worker(request)
