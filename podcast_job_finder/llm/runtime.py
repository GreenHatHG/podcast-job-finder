from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from podcast_job_finder.llm.client import OpenAiCompatibleLlmClient
from podcast_job_finder.llm.config import (
    API_STYLE_ENV_SUFFIX,
    CHAT_COMPLETIONS_API_STYLE,
    build_llm_env_name,
    load_openai_compatible_config_from_env,
)
from podcast_job_finder.llm.errors import OpenAiCompatibleConfigError
from podcast_job_finder.llm.rate_limit import (
    RateLimitedLlmClient,
    load_llm_rate_from_env,
)
from podcast_job_finder.llm.retry import (
    LlmRetryConfig,
    load_llm_retry_config_from_env,
)


AUDIO_TRANSCRIPTION_LLM_ENV_PREFIX: Final = "LLM_AUDIO_TRANSCRIPTION"
TRANSCRIPTION_FORMATTING_LLM_ENV_PREFIX: Final = "LLM_TRANSCRIPTION_FORMATTING"
AUDIO_COMPANY_EXTRACTION_LLM_ENV_PREFIX: Final = "LLM_AUDIO_COMPANY_EXTRACTION"
PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX: Final = "LLM_PAGE_COMPANY_EXTRACTION"
AUDIO_TRANSCRIPTION_API_STYLE_ENV: Final = build_llm_env_name(
    AUDIO_TRANSCRIPTION_LLM_ENV_PREFIX,
    API_STYLE_ENV_SUFFIX,
)
AUDIO_TRANSCRIPTION_API_STYLE_ERROR_TEMPLATE: Final = (
    "环境变量 {env_name} 必须设置为 chat.completions，音频识别仅支持该接口类型。"
)


@dataclass(slots=True, frozen=True)
class LlmRuntime:
    client: RateLimitedLlmClient
    retry_config: LlmRetryConfig
    model: str
    base_url: str | None
    api_style: str


def load_llm_runtime_from_env(env_prefix: str) -> LlmRuntime:
    client_config = load_openai_compatible_config_from_env(env_prefix)
    retry_config = load_llm_retry_config_from_env(env_prefix)
    rate_per_minute = load_llm_rate_from_env(env_prefix)
    return LlmRuntime(
        client=RateLimitedLlmClient(
            OpenAiCompatibleLlmClient(client_config),
            rate_per_minute,
        ),
        retry_config=retry_config,
        model=client_config.model,
        base_url=client_config.base_url,
        api_style=client_config.api_style,
    )


def load_audio_transcription_llm_runtime_from_env() -> LlmRuntime:
    runtime = load_llm_runtime_from_env(AUDIO_TRANSCRIPTION_LLM_ENV_PREFIX)
    if runtime.api_style != CHAT_COMPLETIONS_API_STYLE:
        raise OpenAiCompatibleConfigError(
            AUDIO_TRANSCRIPTION_API_STYLE_ERROR_TEMPLATE.format(
                env_name=AUDIO_TRANSCRIPTION_API_STYLE_ENV
            )
        )
    return runtime


def load_transcription_formatting_llm_runtime_from_env() -> LlmRuntime:
    return load_llm_runtime_from_env(TRANSCRIPTION_FORMATTING_LLM_ENV_PREFIX)
