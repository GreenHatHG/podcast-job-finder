from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Final

from podcast_job_finder.companies.extraction import LlmClientProtocol


LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE_ENV: Final = (
    "LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE"
)
LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE_ENV: Final = (
    "LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE"
)
INVALID_RATE_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 0 的数字。"


class PipelineRateConfigError(ValueError):
    """Raised when the pipeline rate configuration is invalid."""


@dataclass(slots=True, frozen=True)
class PipelineRateConfig:
    producer_rate_per_minute: float | None = None
    consumer_rate_per_minute: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_rate(
            self.producer_rate_per_minute,
            LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE_ENV,
        )
        _validate_optional_rate(
            self.consumer_rate_per_minute,
            LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE_ENV,
        )


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
        rate_limiter: PerMinuteRateLimiter,
    ) -> None:
        self._wrapped_client = wrapped_client
        self._rate_limiter = rate_limiter

    def generate(self, prompt: str) -> str:
        self._rate_limiter.wait_turn()
        return self._wrapped_client.generate(prompt)


def load_pipeline_rate_config_from_env() -> PipelineRateConfig:
    return PipelineRateConfig(
        producer_rate_per_minute=_get_optional_rate_env(
            LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE_ENV
        ),
        consumer_rate_per_minute=_get_optional_rate_env(
            LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE_ENV
        ),
    )


def format_rate(rate_per_minute: float | None) -> str:
    if rate_per_minute is None:
        return "不限速"
    return f"{rate_per_minute}/分钟"


def _get_optional_rate_env(env_name: str) -> float | None:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None
    try:
        parsed_value = float(normalized_value)
    except ValueError as error:
        raise PipelineRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
        ) from error

    _validate_optional_rate(parsed_value, env_name)
    return parsed_value


def _validate_optional_rate(rate_per_minute: float | None, env_name: str) -> None:
    if rate_per_minute is None:
        return
    if not math.isfinite(rate_per_minute) or rate_per_minute <= 0:
        raise PipelineRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
        )
