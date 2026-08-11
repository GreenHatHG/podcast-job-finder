from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar


_EnvValueT = TypeVar("_EnvValueT")


def get_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None

    normalized_value = value.strip()
    return normalized_value or None


def get_optional_env_value(
    name: str,
    converter: Callable[[str], _EnvValueT],
) -> _EnvValueT | None:
    value = get_optional_env(name)
    return None if value is None else converter(value)
