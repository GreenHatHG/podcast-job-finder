from __future__ import annotations

import json
import logging
import os
import sys
from typing import Final

from company_extraction import CompanyExtractionError, extract_companies_from_episode
from extract_xiaoyuzhou_episode import EpisodeParseError, parse_episode_url
from openai_compatible_llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmClient,
    OpenAiCompatibleLlmError,
    load_openai_compatible_config_from_env,
    load_llm_retry_config_from_env,
)


USAGE_TEXT: Final = "用法：python extract_episode_companies.py <episode_url>"
LOG_LEVEL_ENV: Final = "LOG_LEVEL"
DEFAULT_LOG_LEVEL_NAME: Final = "WARNING"


def main() -> int:
    if len(sys.argv) != 2:
        print(USAGE_TEXT, file=sys.stderr)
        return 1

    episode_url = sys.argv[1]
    try:
        _configure_logging()
        llm_config = load_openai_compatible_config_from_env()
        retry_config = load_llm_retry_config_from_env()
        episode = parse_episode_url(episode_url)
        llm_client = OpenAiCompatibleLlmClient(llm_config)
        extraction_result = extract_companies_from_episode(
            episode,
            llm_client,
            retry_config=retry_config,
        )
    except (
        CompanyExtractionError,
        EmptyLlmResponseError,
        EpisodeParseError,
        OpenAiCompatibleConfigError,
        OpenAiCompatibleLlmError,
        ValueError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(
        json.dumps(
            extraction_result.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _configure_logging() -> None:
    logging.basicConfig(
        level=_resolve_log_level(),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _resolve_log_level() -> int:
    raw_log_level = os.getenv(LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL_NAME)
    normalized_log_level = raw_log_level.strip().upper()
    if not normalized_log_level:
        normalized_log_level = DEFAULT_LOG_LEVEL_NAME

    resolved_log_level = getattr(logging, normalized_log_level, None)
    if not isinstance(resolved_log_level, int):
        return logging.WARNING
    return resolved_log_level


if __name__ == "__main__":
    raise SystemExit(main())
