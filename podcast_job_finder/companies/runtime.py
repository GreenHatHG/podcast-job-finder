from __future__ import annotations

import os
import re
from typing import Final

from podcast_job_finder.companies.episode_runner import (
    EpisodeExtractionRuntime,
    build_runtime_signature,
)
from podcast_job_finder.llm import (
    OpenAiCompatibleLlmClient,
    load_llm_retry_config_from_env,
    load_openai_compatible_config_from_env,
)


COMPANY_BLACKLIST_ENV_NAME: Final = "COMPANY_BLACKLIST"
COMPANY_BLACKLIST_SEPARATOR_PATTERN = re.compile(r"[\n,，]+")


def load_extraction_runtime_from_env() -> EpisodeExtractionRuntime:
    llm_config = load_openai_compatible_config_from_env()
    retry_config = load_llm_retry_config_from_env()
    company_blacklist = _load_company_blacklist()
    return EpisodeExtractionRuntime(
        llm_client=OpenAiCompatibleLlmClient(llm_config),
        retry_config=retry_config,
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
