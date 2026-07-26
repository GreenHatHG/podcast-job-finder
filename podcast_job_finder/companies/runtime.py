from __future__ import annotations

import os
import re
from typing import Final

from podcast_job_finder.companies.episode_runner import (
    EpisodeExtractionRuntime,
    build_runtime_signature,
)
from podcast_job_finder.llm import (
    AUDIO_COMPANY_EXTRACTION_LLM_ENV_PREFIX,
    PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX,
    load_llm_runtime_config_from_env,
)


COMPANY_BLACKLIST_ENV_NAME: Final = "COMPANY_BLACKLIST"
COMPANY_BLACKLIST_SEPARATOR_PATTERN = re.compile(r"[\n,，]+")


def load_page_extraction_runtime_from_env() -> EpisodeExtractionRuntime:
    return _load_extraction_runtime_from_env(PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX)


def load_audio_extraction_runtime_from_env() -> EpisodeExtractionRuntime:
    return _load_extraction_runtime_from_env(AUDIO_COMPANY_EXTRACTION_LLM_ENV_PREFIX)


def _load_extraction_runtime_from_env(env_prefix: str) -> EpisodeExtractionRuntime:
    llm_runtime = load_llm_runtime_config_from_env(env_prefix)
    llm_config = llm_runtime.client_config
    company_blacklist = _load_company_blacklist()
    return EpisodeExtractionRuntime(
        llm_client=llm_runtime.build_client(),
        retry_config=llm_runtime.retry_config,
        company_blacklist=company_blacklist,
        model=llm_config.model,
        base_url=llm_config.base_url,
        api_style=llm_config.api_style,
        runtime_signature=build_runtime_signature(
            model=llm_config.model,
            base_url=llm_config.base_url,
            api_style=llm_config.api_style,
            company_blacklist=company_blacklist,
        ),
    )


def _load_company_blacklist() -> tuple[str, ...]:
    normalized_text = os.getenv(COMPANY_BLACKLIST_ENV_NAME, "").strip()
    if not normalized_text:
        return ()
    return tuple(
        company_name.strip()
        for company_name in COMPANY_BLACKLIST_SEPARATOR_PATTERN.split(normalized_text)
        if company_name.strip()
    )
