from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Final

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)

from http_user_agents import DEFAULT_BROWSER_USER_AGENT


OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
OPENAI_MODEL_ENV: Final = "OPENAI_MODEL"
OPENAI_API_STYLE_ENV: Final = "OPENAI_API_STYLE"
OPENAI_BASE_URL_ENV: Final = "OPENAI_BASE_URL"
OPENAI_MAX_ATTEMPTS_ENV: Final = "OPENAI_MAX_ATTEMPTS"
OPENAI_RETRY_BASE_DELAY_SECONDS_ENV: Final = "OPENAI_RETRY_BASE_DELAY_SECONDS"
OPENAI_RETRY_MAX_DELAY_SECONDS_ENV: Final = "OPENAI_RETRY_MAX_DELAY_SECONDS"
RESPONSES_API_STYLE: Final = "responses"
CHAT_COMPLETIONS_API_STYLE: Final = "chat.completions"
SUPPORTED_API_STYLES: Final = (
    RESPONSES_API_STYLE,
    CHAT_COMPLETIONS_API_STYLE,
)
DEFAULT_MAX_ATTEMPTS: Final = 3
DEFAULT_RETRY_BASE_DELAY_SECONDS: Final = 1.0
DEFAULT_RETRY_MAX_DELAY_SECONDS: Final = 8.0
RETRYABLE_STATUS_CODES: Final = frozenset({408, 409, 429})
MISSING_ENV_ERROR_TEMPLATE: Final = "缺少环境变量：{env_name}"
INVALID_API_STYLE_ERROR_TEMPLATE: Final = (
    "环境变量 OPENAI_API_STYLE 仅支持以下取值：responses, chat.completions。"
)
INVALID_INTEGER_ENV_TEMPLATE: Final = (
    "环境变量 {env_name} 必须是大于等于 {minimum} 的整数。"
)
INVALID_FLOAT_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 {minimum} 的数字。"
INVALID_MAX_DELAY_ERROR: Final = (
    "环境变量 OPENAI_RETRY_MAX_DELAY_SECONDS 必须大于等于 "
    "OPENAI_RETRY_BASE_DELAY_SECONDS。"
)
EMPTY_RESPONSE_TEXT_ERROR: Final = "LLM 返回了空文本。"
LLM_REQUEST_ERROR_TEMPLATE: Final = "LLM 调用失败：{error_message}"
USER_ROLE: Final = "user"


class OpenAiCompatibleConfigError(ValueError):
    """Raised when the OpenAI-compatible client configuration is invalid."""


class OpenAiCompatibleLlmError(RuntimeError):
    """Raised when the OpenAI-compatible client request fails."""


class RetryableOpenAiCompatibleLlmError(OpenAiCompatibleLlmError):
    """Raised when the OpenAI-compatible request can be retried."""


class EmptyLlmResponseError(RuntimeError):
    """Raised when the LLM responds with an empty text payload."""


@dataclass(slots=True, frozen=True)
class OpenAiCompatibleConfig:
    api_key: str
    model: str
    api_style: str
    base_url: str | None = None


@dataclass(slots=True, frozen=True)
class LlmRetryConfig:
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    # First retry waits this long, then later retries grow exponentially from it.
    base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS
    # Caps the exponential backoff so each retry wait never exceeds this value.
    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS

    def __post_init__(self) -> None:
        _validate_max_attempts(self.max_attempts)
        _validate_base_delay_seconds(self.base_delay_seconds)
        _validate_max_delay_seconds(
            self.max_delay_seconds,
            self.base_delay_seconds,
        )


class OpenAiCompatibleLlmClient:
    def __init__(self, config: OpenAiCompatibleConfig) -> None:
        _validate_api_style(config.api_style)
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=0,
            default_headers={"User-Agent": DEFAULT_BROWSER_USER_AGENT},
        )

    def generate(self, prompt: str) -> str:
        try:
            if self._config.api_style == RESPONSES_API_STYLE:
                return self._generate_with_responses(prompt)
            return self._generate_with_chat_completions(prompt)
        except OpenAIError as error:
            if _is_retryable_openai_error(error):
                raise RetryableOpenAiCompatibleLlmError(
                    LLM_REQUEST_ERROR_TEMPLATE.format(error_message=str(error))
                ) from error
            raise OpenAiCompatibleLlmError(
                LLM_REQUEST_ERROR_TEMPLATE.format(error_message=str(error))
            ) from error

    def _generate_with_responses(self, prompt: str) -> str:
        response = self._client.responses.create(
            model=self._config.model,
            input=prompt,
        )
        return _normalize_response_text(response.output_text)

    def _generate_with_chat_completions(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": USER_ROLE, "content": prompt}],
        )
        if not response.choices:
            raise EmptyLlmResponseError(EMPTY_RESPONSE_TEXT_ERROR)

        message_content = response.choices[0].message.content
        return _normalize_response_text(message_content)


def load_openai_compatible_config_from_env() -> OpenAiCompatibleConfig:
    api_key = _get_required_env(OPENAI_API_KEY_ENV)
    model = _get_required_env(OPENAI_MODEL_ENV)
    api_style = _get_required_env(OPENAI_API_STYLE_ENV)
    normalized_api_style = api_style.strip()
    _validate_api_style(normalized_api_style)

    base_url = os.getenv(OPENAI_BASE_URL_ENV)
    normalized_base_url = _normalize_optional_env_value(base_url)
    return OpenAiCompatibleConfig(
        api_key=api_key,
        model=model,
        api_style=normalized_api_style,
        base_url=normalized_base_url,
    )


def load_llm_retry_config_from_env() -> LlmRetryConfig:
    max_attempts = _get_integer_env(
        OPENAI_MAX_ATTEMPTS_ENV,
        DEFAULT_MAX_ATTEMPTS,
        minimum=1,
    )
    base_delay_seconds = _get_float_env(
        OPENAI_RETRY_BASE_DELAY_SECONDS_ENV,
        DEFAULT_RETRY_BASE_DELAY_SECONDS,
        minimum=0.0,
    )
    max_delay_seconds = _get_float_env(
        OPENAI_RETRY_MAX_DELAY_SECONDS_ENV,
        DEFAULT_RETRY_MAX_DELAY_SECONDS,
        minimum=0.0,
    )
    return LlmRetryConfig(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        max_delay_seconds=max_delay_seconds,
    )


def _get_required_env(env_name: str) -> str:
    value = os.getenv(env_name)
    normalized_value = _normalize_optional_env_value(value)
    if normalized_value is None:
        raise OpenAiCompatibleConfigError(
            MISSING_ENV_ERROR_TEMPLATE.format(env_name=env_name)
        )
    return normalized_value


def _get_integer_env(env_name: str, default_value: int, minimum: int) -> int:
    raw_value = _normalize_optional_env_value(os.getenv(env_name))
    if raw_value is None:
        return default_value

    try:
        parsed_value = int(raw_value)
    except ValueError as error:
        raise OpenAiCompatibleConfigError(
            INVALID_INTEGER_ENV_TEMPLATE.format(
                env_name=env_name,
                minimum=minimum,
            )
        ) from error

    if parsed_value < minimum:
        raise OpenAiCompatibleConfigError(
            INVALID_INTEGER_ENV_TEMPLATE.format(
                env_name=env_name,
                minimum=minimum,
            )
        )
    return parsed_value


def _get_float_env(env_name: str, default_value: float, minimum: float) -> float:
    raw_value = _normalize_optional_env_value(os.getenv(env_name))
    if raw_value is None:
        return default_value

    try:
        parsed_value = float(raw_value)
    except ValueError as error:
        raise OpenAiCompatibleConfigError(
            INVALID_FLOAT_ENV_TEMPLATE.format(
                env_name=env_name,
                minimum=minimum,
            )
        ) from error

    if not math.isfinite(parsed_value) or parsed_value <= minimum:
        raise OpenAiCompatibleConfigError(
            INVALID_FLOAT_ENV_TEMPLATE.format(
                env_name=env_name,
                minimum=minimum,
            )
        )
    return parsed_value


def _normalize_optional_env_value(value: str | None) -> str | None:
    if value is None:
        return None

    normalized_value = value.strip()
    if not normalized_value:
        return None
    return normalized_value


def _validate_api_style(api_style: str) -> None:
    if api_style not in SUPPORTED_API_STYLES:
        raise OpenAiCompatibleConfigError(INVALID_API_STYLE_ERROR_TEMPLATE)


def _validate_max_attempts(max_attempts: int) -> None:
    if max_attempts < 1:
        raise OpenAiCompatibleConfigError(
            INVALID_INTEGER_ENV_TEMPLATE.format(
                env_name=OPENAI_MAX_ATTEMPTS_ENV,
                minimum=1,
            )
        )


def _validate_base_delay_seconds(base_delay_seconds: float) -> None:
    if not math.isfinite(base_delay_seconds) or base_delay_seconds <= 0:
        raise OpenAiCompatibleConfigError(
            INVALID_FLOAT_ENV_TEMPLATE.format(
                env_name=OPENAI_RETRY_BASE_DELAY_SECONDS_ENV,
                minimum=0.0,
            )
        )


def _validate_max_delay_seconds(
    max_delay_seconds: float,
    base_delay_seconds: float,
) -> None:
    if not math.isfinite(max_delay_seconds) or max_delay_seconds <= 0:
        raise OpenAiCompatibleConfigError(
            INVALID_FLOAT_ENV_TEMPLATE.format(
                env_name=OPENAI_RETRY_MAX_DELAY_SECONDS_ENV,
                minimum=0.0,
            )
        )

    if max_delay_seconds < base_delay_seconds:
        raise OpenAiCompatibleConfigError(INVALID_MAX_DELAY_ERROR)


def _is_retryable_openai_error(error: OpenAIError) -> bool:
    if isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
        ),
    ):
        return True

    if not isinstance(error, APIStatusError):
        return False

    status_code = getattr(error, "status_code", None)
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return isinstance(status_code, int) and 500 <= status_code <= 599


def _normalize_response_text(response_text: str | None) -> str:
    normalized_text = (response_text or "").strip()
    if not normalized_text:
        raise EmptyLlmResponseError(EMPTY_RESPONSE_TEXT_ERROR)
    return normalized_text
