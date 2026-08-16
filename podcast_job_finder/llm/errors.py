from __future__ import annotations

from typing import Final

from podcast_job_finder.errors import ConfigurationError, EpisodeProcessingError


LLM_RETRY_EXHAUSTED_ERROR_TEMPLATE: Final = (
    "{operation_name}连续 {max_attempts} 次尝试失败，最后一次错误：{error_message}"
)


class OpenAiCompatibleConfigError(ConfigurationError, ValueError):
    """Raised when the OpenAI-compatible client configuration is invalid."""


class OpenAiCompatibleLlmError(EpisodeProcessingError, RuntimeError):
    """Raised when the OpenAI-compatible client request fails."""


class RetryableOpenAiCompatibleLlmError(OpenAiCompatibleLlmError):
    """Raised when the OpenAI-compatible request can be retried."""


class LlmRetryExhaustedError(EpisodeProcessingError, RuntimeError):
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


class EmptyLlmResponseError(EpisodeProcessingError, RuntimeError):
    """Raised when the LLM responds with an empty text payload."""
