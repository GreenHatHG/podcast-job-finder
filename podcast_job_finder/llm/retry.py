from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Final, TypeVar

from podcast_job_finder.llm.errors import (
    LlmRetryExhaustedError,
    OpenAiCompatibleConfigError,
    RetryableOpenAiCompatibleLlmError,
)


OPENAI_MAX_ATTEMPTS_ENV: Final = "OPENAI_MAX_ATTEMPTS"
OPENAI_RETRY_BASE_DELAY_SECONDS_ENV: Final = "OPENAI_RETRY_BASE_DELAY_SECONDS"
OPENAI_RETRY_MAX_DELAY_SECONDS_ENV: Final = "OPENAI_RETRY_MAX_DELAY_SECONDS"
DEFAULT_MAX_ATTEMPTS: Final = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS: Final = 1.0
DEFAULT_RETRY_MAX_DELAY_SECONDS: Final = 8.0
DEFAULT_LLM_OPERATION_NAME: Final = "LLM"
INVALID_INTEGER_ENV_TEMPLATE: Final = (
    "环境变量 {env_name} 必须是大于等于 {minimum} 的整数。"
)
INVALID_FLOAT_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 {minimum} 的数字。"
INVALID_MAX_DELAY_ERROR: Final = (
    "环境变量 OPENAI_RETRY_MAX_DELAY_SECONDS 必须大于等于 "
    "OPENAI_RETRY_BASE_DELAY_SECONDS。"
)
_NumericT = TypeVar("_NumericT", int, float)
_RetryResultT = TypeVar("_RetryResultT")

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class LlmRetryConfig:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS
    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS

    def __post_init__(self) -> None:
        _validate_max_attempts(self.max_attempts)
        _validate_delay_seconds(
            self.base_delay_seconds,
            OPENAI_RETRY_BASE_DELAY_SECONDS_ENV,
        )
        _validate_delay_seconds(
            self.max_delay_seconds,
            OPENAI_RETRY_MAX_DELAY_SECONDS_ENV,
        )
        if self.max_delay_seconds < self.base_delay_seconds:
            raise OpenAiCompatibleConfigError(INVALID_MAX_DELAY_ERROR)


DEFAULT_RETRYABLE_LLM_ERRORS: Final[tuple[type[Exception], ...]] = (
    RetryableOpenAiCompatibleLlmError,
)


def execute_llm_with_retry(
    operation: Callable[[], _RetryResultT],
    *,
    retry_config: LlmRetryConfig | None = None,
    retryable_errors: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_LLM_ERRORS,
    operation_name: str = DEFAULT_LLM_OPERATION_NAME,
) -> tuple[_RetryResultT, int]:
    effective_config = retry_config or LlmRetryConfig()
    for attempt in range(1, effective_config.max_attempts + 1):
        _log_attempt(operation_name, attempt)
        try:
            return operation(), attempt
        except retryable_errors as error:
            if attempt == effective_config.max_attempts:
                raise LlmRetryExhaustedError(
                    operation_name=operation_name,
                    max_attempts=effective_config.max_attempts,
                    last_error=error,
                ) from error
            _wait_before_retry(operation_name, attempt, error, effective_config)

    raise AssertionError("LLM 重试循环未返回结果。")


def load_llm_retry_config_from_env() -> LlmRetryConfig:
    return LlmRetryConfig(
        max_attempts=_get_integer_env(
            OPENAI_MAX_ATTEMPTS_ENV,
            DEFAULT_MAX_ATTEMPTS,
            minimum=1,
        ),
        base_delay_seconds=_get_float_env(
            OPENAI_RETRY_BASE_DELAY_SECONDS_ENV,
            DEFAULT_RETRY_BASE_DELAY_SECONDS,
        ),
        max_delay_seconds=_get_float_env(
            OPENAI_RETRY_MAX_DELAY_SECONDS_ENV,
            DEFAULT_RETRY_MAX_DELAY_SECONDS,
        ),
    )


def _get_integer_env(env_name: str, default_value: int, *, minimum: int) -> int:
    raw_value = _get_optional_env(env_name)
    if raw_value is None:
        return default_value
    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise _build_integer_config_error(env_name, minimum) from error
    if parsed_value < minimum:
        raise _build_integer_config_error(env_name, minimum)
    return parsed_value


def _get_float_env(env_name: str, default_value: float) -> float:
    raw_value = _get_optional_env(env_name)
    if raw_value is None:
        return default_value
    try:
        parsed_value = float(raw_value)
    except ValueError as error:
        raise _build_float_config_error(env_name) from error
    _validate_delay_seconds(parsed_value, env_name)
    return parsed_value


def _get_optional_env(env_name: str) -> str | None:
    value = os.getenv(env_name)
    if value is None:
        return None
    normalized_value = value.strip()
    return normalized_value or None


def _validate_max_attempts(max_attempts: int) -> None:
    if max_attempts < 1:
        raise _build_integer_config_error(OPENAI_MAX_ATTEMPTS_ENV, 1)


def _validate_delay_seconds(delay_seconds: float, env_name: str) -> None:
    if not math.isfinite(delay_seconds) or delay_seconds <= 0:
        raise _build_float_config_error(env_name)


def _build_integer_config_error(
    env_name: str,
    minimum: int,
) -> OpenAiCompatibleConfigError:
    return OpenAiCompatibleConfigError(
        INVALID_INTEGER_ENV_TEMPLATE.format(env_name=env_name, minimum=minimum)
    )


def _build_float_config_error(env_name: str) -> OpenAiCompatibleConfigError:
    return OpenAiCompatibleConfigError(
        INVALID_FLOAT_ENV_TEMPLATE.format(env_name=env_name, minimum=0.0)
    )


def _log_attempt(operation_name: str, attempt: int) -> None:
    if attempt == 1:
        logger.info("%s 调用中...", operation_name)
        return
    logger.info("%s 重试中...（第 %d 次）", operation_name, attempt)


def _wait_before_retry(
    operation_name: str,
    attempt: int,
    error: Exception,
    config: LlmRetryConfig,
) -> None:
    delay_seconds = min(
        config.base_delay_seconds * (2 ** (attempt - 1)),
        config.max_delay_seconds,
    )
    logger.debug(
        "%s 第 %s 次尝试失败，将在 %.2f 秒后重试。错误：%s",
        operation_name,
        attempt,
        delay_seconds,
        error,
    )
    time.sleep(delay_seconds)
