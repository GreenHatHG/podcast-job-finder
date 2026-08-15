from __future__ import annotations

import math
import threading
import time
from typing import Final, Protocol

from podcast_job_finder.environment import get_optional_env_value
from podcast_job_finder.llm.client import AudioFormat
from podcast_job_finder.llm.config import build_llm_env_name
from podcast_job_finder.llm.errors import OpenAiCompatibleConfigError


RATE_PER_MINUTE_ENV_SUFFIX: Final = "RATE_PER_MINUTE"
MAX_IN_FLIGHT_REQUESTS_ENV_SUFFIX: Final = "MAX_IN_FLIGHT_REQUESTS"
DEFAULT_MAX_IN_FLIGHT_REQUESTS: Final = 10
INVALID_RATE_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 0 的数字。"
INVALID_MAX_IN_FLIGHT_REQUESTS_ENV_TEMPLATE: Final = (
    "环境变量 {env_name} 必须是大于 0 的整数。"
)


class LlmClientProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...

    def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str: ...


class PerMinuteRateLimiter:
    def __init__(self, rate_per_minute: float | None) -> None:
        self._min_interval_seconds = (
            None if rate_per_minute is None else 60.0 / rate_per_minute
        )
        self._next_allowed_at: float | None = None

    def wait_turn(self) -> None:
        if self._min_interval_seconds is None:
            return

        current_time = time.monotonic()
        if self._next_allowed_at is None:
            self._next_allowed_at = current_time + self._min_interval_seconds
            return

        sleep_seconds = self._next_allowed_at - current_time
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
            current_time = self._next_allowed_at
        else:
            current_time = time.monotonic()
        self._next_allowed_at = current_time + self._min_interval_seconds


class LlmRequestGate:
    """限制同一个 LLM 客户端的请求开始频率和同时执行数量。

    多个调用线程共享这个对象。Condition 内部的锁保证同一时刻只有一个线程检查或
    修改请求数量和下次允许开始请求的时间。线程调用 wait() 后会释放这把锁并暂停；
    收到通知或等待超时后，它会先重新取得锁，wait() 才会返回，因此后续读取仍受锁
    保护。
    """

    def __init__(
        self,
        rate_per_minute: float | None,
        max_in_flight_requests: int,
    ) -> None:
        if max_in_flight_requests <= 0:
            raise ValueError("max_in_flight_requests 必须大于 0。")
        self._min_interval_seconds = (
            None if rate_per_minute is None else 60.0 / rate_per_minute
        )
        self._max_in_flight_requests = max_in_flight_requests
        self._in_flight_requests = 0
        self._next_allowed_at: float | None = None
        self._condition = threading.Condition()

    def enter(self) -> None:
        with self._condition:
            # 线程被唤醒不代表已经获得请求名额。其他线程可能先一步修改状态，所以
            # 每次 wait() 返回后都要从头检查请求数量和允许开始的时间。
            while True:
                if self._in_flight_requests >= self._max_in_flight_requests:
                    # 等待期间会释放 Condition 内部的锁，让已经完成请求的线程能够
                    # 进入 leave() 归还名额。重新取得锁后，wait() 才会返回。
                    self._condition.wait()
                    continue

                current_time = time.monotonic()
                if (
                    self._next_allowed_at is not None
                    and current_time < self._next_allowed_at
                ):
                    # 使用带超时的 wait() 等到允许开始的时间。等待时同样会释放锁；
                    # 即使被 leave() 提前唤醒，也会回到循环重新检查时间。
                    self._condition.wait(self._next_allowed_at - current_time)
                    continue

                # 在仍持有锁时占用名额并安排下一个开始时间，防止另一个线程根据同一份
                # 旧状态同时通过检查。离开 with 后才释放锁并执行实际网络请求。
                self._in_flight_requests += 1
                if self._min_interval_seconds is not None:
                    self._next_allowed_at = current_time + self._min_interval_seconds
                return

    def leave(self) -> None:
        with self._condition:
            self._in_flight_requests -= 1
            # 通知所有等待线程状态已经变化。被唤醒的线程不会直接开始请求，而是要在
            # 当前线程释放锁后逐个取得锁，并重新检查 enter() 中的两个条件。
            self._condition.notify_all()


class RateLimitedLlmClient:
    def __init__(
        self,
        wrapped_client: LlmClientProtocol,
        rate_per_minute: float | None,
        max_in_flight_requests: int = DEFAULT_MAX_IN_FLIGHT_REQUESTS,
    ) -> None:
        self._wrapped_client = wrapped_client
        self._request_gate = LlmRequestGate(
            rate_per_minute,
            max_in_flight_requests,
        )

    def generate(self, prompt: str) -> str:
        self._request_gate.enter()
        try:
            return self._wrapped_client.generate(prompt)
        finally:
            self._request_gate.leave()

    def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str:
        self._request_gate.enter()
        try:
            return self._wrapped_client.transcribe_audio(
                audio_data,
                audio_format=audio_format,
                prompt=prompt,
            )
        finally:
            self._request_gate.leave()


def load_llm_rate_from_env(env_prefix: str) -> float | None:
    env_name = build_llm_env_name(env_prefix, RATE_PER_MINUTE_ENV_SUFFIX)
    try:
        rate_per_minute = get_optional_env_value(env_name, float)
    except ValueError as error:
        raise _build_rate_config_error(env_name) from error
    _validate_rate(rate_per_minute, env_name)
    return rate_per_minute


def load_llm_max_in_flight_requests_from_env(env_prefix: str) -> int:
    env_name = build_llm_env_name(
        env_prefix,
        MAX_IN_FLIGHT_REQUESTS_ENV_SUFFIX,
    )
    try:
        max_in_flight_requests = get_optional_env_value(env_name, int)
    except ValueError as error:
        raise _build_max_in_flight_requests_config_error(env_name) from error
    if max_in_flight_requests is None:
        return DEFAULT_MAX_IN_FLIGHT_REQUESTS
    if max_in_flight_requests <= 0:
        raise _build_max_in_flight_requests_config_error(env_name)
    return max_in_flight_requests


def format_rate(rate_per_minute: float | None) -> str:
    if rate_per_minute is None:
        return "不限速"
    return f"{rate_per_minute}/分钟"


def _validate_rate(rate_per_minute: float | None, env_name: str) -> None:
    if rate_per_minute is None:
        return
    if not math.isfinite(rate_per_minute) or rate_per_minute <= 0:
        raise _build_rate_config_error(env_name)


def _build_rate_config_error(env_name: str) -> OpenAiCompatibleConfigError:
    return OpenAiCompatibleConfigError(
        INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
    )


def _build_max_in_flight_requests_config_error(
    env_name: str,
) -> OpenAiCompatibleConfigError:
    return OpenAiCompatibleConfigError(
        INVALID_MAX_IN_FLIGHT_REQUESTS_ENV_TEMPLATE.format(env_name=env_name)
    )
