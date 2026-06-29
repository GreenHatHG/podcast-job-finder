from __future__ import annotations

import json
import sys
from typing import Final

from company_extraction import CompanyExtractionError, extract_companies_from_episode
from extract_xiaoyuzhou_episode import EpisodeParseError, parse_episode_url
from openai_compatible_llm import (
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmClient,
    OpenAiCompatibleLlmError,
    load_openai_compatible_config_from_env,
)


USAGE_TEXT: Final = "用法：python extract_episode_companies.py <episode_url>"


def main() -> int:
    if len(sys.argv) != 2:
        print(USAGE_TEXT, file=sys.stderr)
        return 1

    episode_url = sys.argv[1]
    try:
        episode = parse_episode_url(episode_url)
        llm_client = OpenAiCompatibleLlmClient(
            load_openai_compatible_config_from_env()
        )
        extraction_result = extract_companies_from_episode(episode, llm_client)
    except (
        CompanyExtractionError,
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


if __name__ == "__main__":
    raise SystemExit(main())
