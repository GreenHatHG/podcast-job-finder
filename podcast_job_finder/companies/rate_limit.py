from __future__ import annotations

import math
from typing import Final

from podcast_job_finder.environment import get_optional_env_value
from podcast_job_finder.errors import ConfigurationError


EPISODE_PAGE_FETCH_RATE_PER_MINUTE_ENV: Final = "EPISODE_PAGE_FETCH_RATE_PER_MINUTE"
INVALID_RATE_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 0 的数字。"


class EpisodePageFetchRateConfigError(ConfigurationError, ValueError):
    """单集页面请求速率配置无效。"""


def load_episode_page_fetch_rate_from_env() -> float | None:
    try:
        rate_per_minute = get_optional_env_value(
            EPISODE_PAGE_FETCH_RATE_PER_MINUTE_ENV,
            float,
        )
    except ValueError as error:
        raise EpisodePageFetchRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(
                env_name=EPISODE_PAGE_FETCH_RATE_PER_MINUTE_ENV
            )
        ) from error
    if rate_per_minute is not None and (
        not math.isfinite(rate_per_minute) or rate_per_minute <= 0
    ):
        raise EpisodePageFetchRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(
                env_name=EPISODE_PAGE_FETCH_RATE_PER_MINUTE_ENV
            )
        )
    return rate_per_minute
