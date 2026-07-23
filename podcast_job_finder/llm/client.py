from __future__ import annotations

from base64 import b64encode
from typing import Callable, Final, Literal

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
from podcast_job_finder.llm.config import (
    CHAT_COMPLETIONS_API_STYLE,
    RESPONSES_API_STYLE,
    OpenAiCompatibleConfig,
    validate_api_style,
)
from podcast_job_finder.llm.errors import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
    RetryableOpenAiCompatibleLlmError,
)


RETRYABLE_STATUS_CODES: Final = frozenset({408, 409, 429})
EMPTY_RESPONSE_TEXT_ERROR: Final = "LLM 返回了空文本。"
LLM_REQUEST_ERROR_TEMPLATE: Final = "LLM 调用失败：{error_message}"
AUDIO_REQUIRES_CHAT_COMPLETIONS_ERROR: Final = (
    "音频识别仅支持 chat.completions API 风格。"
)
EMPTY_AUDIO_ERROR: Final = "待识别音频不能为空。"
USER_ROLE: Final = "user"
AudioFormat = Literal["wav", "mp3"]


class OpenAiCompatibleLlmClient:
    def __init__(self, config: OpenAiCompatibleConfig) -> None:
        validate_api_style(config.api_style)
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


def _extract_chat_completion_text(response: ChatCompletion) -> str:
    if not response.choices:
        raise EmptyLlmResponseError(EMPTY_RESPONSE_TEXT_ERROR)
    return _normalize_response_text(response.choices[0].message.content)
