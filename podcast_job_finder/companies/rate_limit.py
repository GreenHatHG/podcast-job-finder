from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Final

from podcast_job_finder.llm.rate_limit import load_llm_rate_from_env
from podcast_job_finder.llm.runtime import PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX


EPISODE_PAGE_FETCH_RATE_PER_MINUTE_ENV: Final = "EPISODE_PAGE_FETCH_RATE_PER_MINUTE"
INVALID_RATE_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 0 的数字。"


class PipelineRateConfigError(ValueError):
    """Raised when the pipeline rate configuration is invalid."""


@dataclass(slots=True, frozen=True)
class PipelineRateConfig:
    producer_rate_per_minute: float | None = None
    consumer_rate_per_minute: float | None = None


def load_pipeline_rate_config_from_env() -> PipelineRateConfig:
    return PipelineRateConfig(
        producer_rate_per_minute=_get_optional_rate_env(
            EPISODE_PAGE_FETCH_RATE_PER_MINUTE_ENV
        ),
        consumer_rate_per_minute=load_llm_rate_from_env(
            PAGE_COMPANY_EXTRACTION_LLM_ENV_PREFIX
        ),
    )


def _get_optional_rate_env(env_name: str) -> float | None:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None
    try:
        parsed_value = float(normalized_value)
    except ValueError as error:
        raise PipelineRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
        ) from error

    _validate_optional_rate(parsed_value, env_name)
    return parsed_value


def _validate_optional_rate(
    rate_per_minute: float | None,
    env_name: str,
) -> None:
    if rate_per_minute is None:
        return
    if not math.isfinite(rate_per_minute) or rate_per_minute <= 0:
        raise PipelineRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
        )
