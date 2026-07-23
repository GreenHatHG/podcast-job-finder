from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from podcast_job_finder.llm.errors import OpenAiCompatibleConfigError


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


@dataclass(slots=True, frozen=True)
class OpenAiCompatibleConfig:
    api_key: str
    model: str
    api_style: str
    base_url: str | None = None


def load_openai_compatible_config_from_env() -> OpenAiCompatibleConfig:
    api_key = _get_required_env(OPENAI_API_KEY_ENV)
    model = _get_required_env(OPENAI_MODEL_ENV)
    api_style = _get_required_env(OPENAI_API_STYLE_ENV)
    normalized_api_style = api_style.strip()
    validate_api_style(normalized_api_style)

    base_url = os.getenv(OPENAI_BASE_URL_ENV)
    return OpenAiCompatibleConfig(
        api_key=api_key,
        model=model,
        api_style=normalized_api_style,
        base_url=_normalize_optional_env_value(base_url),
    )


def validate_api_style(api_style: str) -> None:
    if api_style not in SUPPORTED_API_STYLES:
        raise OpenAiCompatibleConfigError(INVALID_API_STYLE_ERROR_TEMPLATE)


def _get_required_env(env_name: str) -> str:
    normalized_value = _normalize_optional_env_value(os.getenv(env_name))
    if normalized_value is None:
        raise OpenAiCompatibleConfigError(
            MISSING_ENV_ERROR_TEMPLATE.format(env_name=env_name)
        )
    return normalized_value


def _normalize_optional_env_value(value: str | None) -> str | None:
    if value is None:
        return None

    normalized_value = value.strip()
    return normalized_value or None
