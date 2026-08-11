from __future__ import annotations

import re
from typing import Final

from podcast_job_finder.companies.episode_runner import EpisodeExtractionRuntime
from podcast_job_finder.companies.extraction import PROMPT_TEMPLATE
from podcast_job_finder.environment import get_optional_env
from podcast_job_finder.llm import (
    AUDIO_COMPANY_EXTRACTION_LLM_ENV_PREFIX,
    LlmRuntime,
    PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX,
    load_llm_runtime_from_env,
)
from podcast_job_finder.runtime_signature import build_runtime_signature_hash


COMPANY_BLACKLIST_ENV_NAME: Final = "COMPANY_BLACKLIST"
COMPANY_BLACKLIST_SEPARATOR_PATTERN = re.compile(r"[\n,，]+")


def load_page_extraction_runtime_from_env() -> EpisodeExtractionRuntime:
    return _load_extraction_runtime_from_env(PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX)


def load_audio_extraction_runtime_from_env() -> EpisodeExtractionRuntime:
    return _load_extraction_runtime_from_env(AUDIO_COMPANY_EXTRACTION_LLM_ENV_PREFIX)


def _load_extraction_runtime_from_env(env_prefix: str) -> EpisodeExtractionRuntime:
    llm_runtime = load_llm_runtime_from_env(env_prefix)
    company_blacklist = _load_company_blacklist()
    return EpisodeExtractionRuntime(
        llm=llm_runtime,
        company_blacklist=company_blacklist,
        runtime_signature=_build_runtime_signature(
            llm=llm_runtime,
            company_blacklist=company_blacklist,
        ),
    )


def _build_runtime_signature(
    *,
    llm: LlmRuntime,
    company_blacklist: tuple[str, ...],
) -> str:
    normalized_blacklist = sorted(
        {
            company_name.strip().casefold()
            for company_name in company_blacklist
            if company_name.strip()
        }
    )
    return build_runtime_signature_hash(
        {
            "model": llm.model,
            "base_url": llm.base_url,
            "api_style": llm.api_style,
            "company_blacklist": normalized_blacklist,
            "prompt_template": PROMPT_TEMPLATE,
        }
    )


def _load_company_blacklist() -> tuple[str, ...]:
    normalized_text = get_optional_env(COMPANY_BLACKLIST_ENV_NAME)
    if normalized_text is None:
        return ()
    return tuple(
        company_name.strip()
        for company_name in COMPANY_BLACKLIST_SEPARATOR_PATTERN.split(normalized_text)
        if company_name.strip()
    )
