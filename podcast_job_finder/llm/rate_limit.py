from __future__ import annotations

import math
import time
from typing import Final, Protocol

from podcast_job_finder.environment import get_optional_env_value
from podcast_job_finder.llm.client import AudioFormat
from podcast_job_finder.llm.config import build_llm_env_name
from podcast_job_finder.llm.errors import OpenAiCompatibleConfigError


RATE_PER_MINUTE_ENV_SUFFIX: Final = "RATE_PER_MINUTE"
INVALID_RATE_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 0 的数字。"


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


class RateLimitedLlmClient:
    def __init__(
        self,
        wrapped_client: LlmClientProtocol,
        rate_per_minute: float | None,
    ) -> None:
        self._wrapped_client = wrapped_client
        self._rate_limiter = PerMinuteRateLimiter(rate_per_minute)

    def generate(self, prompt: str) -> str:
        self._rate_limiter.wait_turn()
        return self._wrapped_client.generate(prompt)

    def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str:
        self._rate_limiter.wait_turn()
        return self._wrapped_client.transcribe_audio(
            audio_data,
            audio_format=audio_format,
            prompt=prompt,
        )


def load_llm_rate_from_env(env_prefix: str) -> float | None:
    env_name = build_llm_env_name(env_prefix, RATE_PER_MINUTE_ENV_SUFFIX)
    try:
        rate_per_minute = get_optional_env_value(env_name, float)
    except ValueError as error:
        raise _build_rate_config_error(env_name) from error
    _validate_rate(rate_per_minute, env_name)
    return rate_per_minute


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
