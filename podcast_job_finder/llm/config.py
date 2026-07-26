from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from podcast_job_finder.llm.errors import OpenAiCompatibleConfigError


API_KEY_ENV_SUFFIX: Final = "API_KEY"
MODEL_ENV_SUFFIX: Final = "MODEL"
API_STYLE_ENV_SUFFIX: Final = "API_STYLE"
BASE_URL_ENV_SUFFIX: Final = "BASE_URL"
RESPONSES_API_STYLE: Final = "responses"
CHAT_COMPLETIONS_API_STYLE: Final = "chat.completions"
SUPPORTED_API_STYLES: Final = (
    RESPONSES_API_STYLE,
    CHAT_COMPLETIONS_API_STYLE,
)
MISSING_ENV_ERROR_TEMPLATE: Final = "缺少环境变量：{env_name}"
INVALID_API_STYLE_ERROR_TEMPLATE: Final = (
    "环境变量 {env_name} 仅支持以下取值：responses, chat.completions。"
)


@dataclass(slots=True, frozen=True)
class OpenAiCompatibleConfig:
    api_key: str
    model: str
    api_style: str
    base_url: str | None = None


def load_openai_compatible_config_from_env(
    env_prefix: str,
) -> OpenAiCompatibleConfig:
    api_key_env = build_llm_env_name(env_prefix, API_KEY_ENV_SUFFIX)
    model_env = build_llm_env_name(env_prefix, MODEL_ENV_SUFFIX)
    api_style_env = build_llm_env_name(env_prefix, API_STYLE_ENV_SUFFIX)
    base_url_env = build_llm_env_name(env_prefix, BASE_URL_ENV_SUFFIX)
    api_key = _get_required_env(api_key_env)
    model = _get_required_env(model_env)
    api_style = _get_required_env(api_style_env)
    normalized_api_style = api_style.strip()
    _validate_api_style(normalized_api_style, api_style_env)

    base_url = os.getenv(base_url_env)
    return OpenAiCompatibleConfig(
        api_key=api_key,
        model=model,
        api_style=normalized_api_style,
        base_url=_normalize_optional_env_value(base_url),
    )


def _validate_api_style(api_style: str, env_name: str) -> None:
    if api_style not in SUPPORTED_API_STYLES:
        raise OpenAiCompatibleConfigError(
            INVALID_API_STYLE_ERROR_TEMPLATE.format(env_name=env_name)
        )


def build_llm_env_name(env_prefix: str, suffix: str) -> str:
    return f"{env_prefix}_{suffix}"


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
