from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from openai import OpenAI, OpenAIError


OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
OPENAI_MODEL_ENV: Final = "OPENAI_MODEL"
OPENAI_API_STYLE_ENV: Final = "OPENAI_API_STYLE"
OPENAI_BASE_URL_ENV: Final = "OPENAI_BASE_URL"
RESPONSES_API_STYLE: Final = "responses"
CHAT_COMPLETIONS_API_STYLE: Final = "chat.completions"
SUPPORTED_API_STYLES: Final = (
    RESPONSES_API_STYLE,
    CHAT_COMPLETIONS_API_STYLE,
)
MISSING_ENV_ERROR_TEMPLATE: Final = "缺少环境变量：{env_name}"
INVALID_API_STYLE_ERROR_TEMPLATE: Final = (
    "环境变量 OPENAI_API_STYLE 仅支持以下取值：responses, chat.completions。"
)
EMPTY_RESPONSE_TEXT_ERROR: Final = "LLM 返回了空文本。"
LLM_REQUEST_ERROR_TEMPLATE: Final = "LLM 调用失败：{error_message}"
USER_ROLE: Final = "user"


class OpenAiCompatibleConfigError(ValueError):
    """Raised when the OpenAI-compatible client configuration is invalid."""


class OpenAiCompatibleLlmError(RuntimeError):
    """Raised when the OpenAI-compatible client request fails."""


@dataclass(slots=True, frozen=True)
class OpenAiCompatibleConfig:
    api_key: str
    model: str
    api_style: str
    base_url: str | None = None


class OpenAiCompatibleLlmClient:
    def __init__(self, config: OpenAiCompatibleConfig) -> None:
        _validate_api_style(config.api_style)
        self._config = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def generate(self, prompt: str) -> str:
        try:
            if self._config.api_style == RESPONSES_API_STYLE:
                return self._generate_with_responses(prompt)
            return self._generate_with_chat_completions(prompt)
        except OpenAIError as error:
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
            raise OpenAiCompatibleLlmError(EMPTY_RESPONSE_TEXT_ERROR)

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


def _get_required_env(env_name: str) -> str:
    value = os.getenv(env_name)
    normalized_value = _normalize_optional_env_value(value)
    if normalized_value is None:
        raise OpenAiCompatibleConfigError(
            MISSING_ENV_ERROR_TEMPLATE.format(env_name=env_name)
        )
    return normalized_value


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


def _normalize_response_text(response_text: str | None) -> str:
    normalized_text = (response_text or "").strip()
    if not normalized_text:
        raise OpenAiCompatibleLlmError(EMPTY_RESPONSE_TEXT_ERROR)
    return normalized_text
