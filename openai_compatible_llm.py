from __future__ import annotations

from base64 import b64encode
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Final, Literal, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionUserMessageParam,
)

from podcast_job_finder.http.user_agents import DEFAULT_BROWSER_USER_AGENT


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
AUDIO_REQUIRES_CHAT_COMPLETIONS_ERROR: Final = (
    "音频识别仅支持 chat.completions API 风格。"
)
EMPTY_AUDIO_ERROR: Final = "待识别音频不能为空。"
DEFAULT_LLM_OPERATION_NAME: Final = "LLM"
LLM_RETRY_EXHAUSTED_ERROR_TEMPLATE: Final = (
    "{operation_name}连续 {max_attempts} 次尝试失败，最后一次错误：{error_message}"
)
USER_ROLE: Final = "user"
_NumericT = TypeVar("_NumericT", int, float)
_RetryResultT = TypeVar("_RetryResultT")
AudioFormat = Literal["wav", "mp3"]

logger = logging.getLogger(__name__)


class OpenAiCompatibleConfigError(ValueError):
    """Raised when the OpenAI-compatible client configuration is invalid."""


class OpenAiCompatibleLlmError(RuntimeError):
    """Raised when the OpenAI-compatible client request fails."""


class RetryableOpenAiCompatibleLlmError(OpenAiCompatibleLlmError):
    """Raised when the OpenAI-compatible request can be retried."""


class LlmRetryExhaustedError(RuntimeError):
    """Raised after an LLM operation exhausts all retry attempts."""

    def __init__(
        self,
        *,
        operation_name: str,
        max_attempts: int,
        last_error: Exception,
    ) -> None:
        self.max_attempts = max_attempts
        self.last_error = last_error
        super().__init__(
            LLM_RETRY_EXHAUSTED_ERROR_TEMPLATE.format(
                operation_name=operation_name,
                max_attempts=max_attempts,
                error_message=str(last_error),
            )
        )


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
    effective_retry_config = retry_config or LlmRetryConfig()
    for attempt in range(1, effective_retry_config.max_attempts + 1):
        _log_llm_attempt(operation_name, attempt)
        try:
            return operation(), attempt
        except retryable_errors as error:
            if attempt == effective_retry_config.max_attempts:
                raise LlmRetryExhaustedError(
                    operation_name=operation_name,
                    max_attempts=effective_retry_config.max_attempts,
                    last_error=error,
                ) from error
            _wait_before_llm_retry(
                operation_name=operation_name,
                attempt=attempt,
                error=error,
                retry_config=effective_retry_config,
            )

    raise AssertionError("LLM 重试循环未返回结果。")


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
        request = (
            self._generate_with_responses
            if self._config.api_style == RESPONSES_API_STYLE
            else self._generate_with_chat_completions
        )
        return self._execute_request(lambda: request(prompt))

    def transcribe_audio(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str:
        if self._config.api_style != CHAT_COMPLETIONS_API_STYLE:
            raise OpenAiCompatibleConfigError(AUDIO_REQUIRES_CHAT_COMPLETIONS_ERROR)
        if not audio_data:
            raise ValueError(EMPTY_AUDIO_ERROR)
        return self._execute_request(
            lambda: self._transcribe_audio_with_chat_completions(
                audio_data,
                audio_format=audio_format,
                prompt=prompt,
            )
        )

    def _execute_request(self, request: Callable[[], str]) -> str:
        try:
            return request()
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
        return _extract_chat_completion_text(response)

    def _transcribe_audio_with_chat_completions(
        self,
        audio_data: bytes,
        *,
        audio_format: AudioFormat,
        prompt: str,
    ) -> str:
        text_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": prompt,
        }
        audio_part: ChatCompletionContentPartInputAudioParam = {
            "type": "input_audio",
            "input_audio": {
                "data": b64encode(audio_data).decode("ascii"),
                "format": audio_format,
            },
        }
        message: ChatCompletionUserMessageParam = {
            "role": USER_ROLE,
            "content": [text_part, audio_part],
        }
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[message],
        )
        return _extract_chat_completion_text(response)


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
    parsed_value = _parse_optional_env_number(
        env_name=env_name,
        default_value=default_value,
        parser=int,
        error_template=INVALID_INTEGER_ENV_TEMPLATE,
        minimum=minimum,
    )

    if parsed_value < minimum:
        raise OpenAiCompatibleConfigError(
            INVALID_INTEGER_ENV_TEMPLATE.format(
                env_name=env_name,
                minimum=minimum,
            )
        )
    return parsed_value


def _get_float_env(env_name: str, default_value: float, minimum: float) -> float:
    parsed_value = _parse_optional_env_number(
        env_name=env_name,
        default_value=default_value,
        parser=float,
        error_template=INVALID_FLOAT_ENV_TEMPLATE,
        minimum=minimum,
    )

    if not math.isfinite(parsed_value) or parsed_value <= minimum:
        raise OpenAiCompatibleConfigError(
            INVALID_FLOAT_ENV_TEMPLATE.format(
                env_name=env_name,
                minimum=minimum,
            )
        )
    return parsed_value


def _parse_optional_env_number(
    *,
    env_name: str,
    default_value: _NumericT,
    parser: Callable[[str], _NumericT],
    error_template: str,
    minimum: int | float,
) -> _NumericT:
    raw_value = _normalize_optional_env_value(os.getenv(env_name))
    if raw_value is None:
        return default_value

    try:
        return parser(raw_value)
    except ValueError as error:
        raise OpenAiCompatibleConfigError(
            error_template.format(
                env_name=env_name,
                minimum=minimum,
            )
        ) from error


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


def _log_llm_attempt(operation_name: str, attempt: int) -> None:
    if attempt == 1:
        logger.info("%s 调用中...", operation_name)
        return
    logger.info("%s 重试中...（第 %d 次）", operation_name, attempt)


def _wait_before_llm_retry(
    *,
    operation_name: str,
    attempt: int,
    error: Exception,
    retry_config: LlmRetryConfig,
) -> None:
    delay_seconds = _calculate_retry_delay(
        attempt,
        retry_config.base_delay_seconds,
        retry_config.max_delay_seconds,
    )
    logger.debug(
        "%s 第 %s 次尝试失败，将在 %.2f 秒后重试。错误：%s",
        operation_name,
        attempt,
        delay_seconds,
        error,
    )
    time.sleep(delay_seconds)


def _calculate_retry_delay(
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> float:
    retry_delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
    return min(retry_delay_seconds, max_delay_seconds)


def _normalize_response_text(response_text: str | None) -> str:
    normalized_text = (response_text or "").strip()
    if not normalized_text:
        raise EmptyLlmResponseError(EMPTY_RESPONSE_TEXT_ERROR)
    return normalized_text


def _extract_chat_completion_text(response: ChatCompletion) -> str:
    if not response.choices:
        raise EmptyLlmResponseError(EMPTY_RESPONSE_TEXT_ERROR)
    return _normalize_response_text(response.choices[0].message.content)
