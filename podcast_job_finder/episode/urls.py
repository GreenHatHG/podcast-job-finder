from __future__ import annotations

import re
from typing import Final
from urllib.parse import urlparse


EPISODE_ID_PATTERN = re.compile(r"^[0-9A-Za-z]{24}$")
EPISODE_PATH_PREFIX: Final = "/episode/"
EPISODE_URL_TEMPLATE: Final = "https://www.xiaoyuzhoufm.com/episode/{eid}"


def extract_episode_id_from_url(episode_url: str) -> str | None:
    if not episode_url.startswith(("http://", "https://")):
        return None

    normalized_path = urlparse(episode_url).path.rstrip("/")
    if not normalized_path.startswith(EPISODE_PATH_PREFIX):
        return None

    episode_id = normalized_path.removeprefix(EPISODE_PATH_PREFIX).strip()
    if EPISODE_ID_PATTERN.fullmatch(episode_id) is None:
        return None
    return episode_id


def build_episode_url(eid: str) -> str:
    return EPISODE_URL_TEMPLATE.format(eid=eid)
