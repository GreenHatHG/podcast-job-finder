from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Final

from podcast_job_finder.environment import get_optional_env
from podcast_job_finder.errors import ConfigurationError


FIRERED_PYTHON_ENV: Final = "FIRERED_PYTHON"
FIRERED_ASR_MODEL_DIR_ENV: Final = "FIRERED_ASR_MODEL_DIR"
FIRERED_PUNC_MODEL_DIR_ENV: Final = "FIRERED_PUNC_MODEL_DIR"
FIRERED_ORT_PROVIDER_ENV: Final = "FIRERED_ORT_PROVIDER"
FIRERED_ORT_INTRA_OP_THREADS_ENV: Final = "FIRERED_ORT_INTRA_OP_THREADS"
INVALID_INTEGER_ENV_ERROR: Final = "{name} 必须是整数：{value}"
INVALID_FLOAT_ENV_ERROR: Final = "{name} 必须是有限数值：{value}"
FIRERED_PYTHON_NOT_FOUND_ERROR: Final = (
    "未找到 FireRed Python 解释器，已检查：{candidates}。"
    "请创建对应的虚拟环境或设置 {environment_variable}。"
)
SOURCE_PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
FIRERED_PYTHON_RELATIVE_PATH: Final = Path("firered_worker/.venv/bin/python")
FIRERED_ASR_MODEL_RELATIVE_PATH: Final = Path(
    "models/firered/FireRedASR2-AED-ONNX-modelscope"
)
FIRERED_PUNC_MODEL_RELATIVE_PATH: Final = Path("models/firered/FireRedPunc")


class AudioTranscriptionConfigError(ConfigurationError, ValueError):
    """音频转写后端配置无效。"""


def load_firered_python() -> Path:
    configured_path = get_optional_env(FIRERED_PYTHON_ENV)
    if configured_path is not None:
        return Path(configured_path)
    candidates = _candidate_project_paths(FIRERED_PYTHON_RELATIVE_PATH)
    for python_executable in candidates:
        if python_executable.is_file():
            return python_executable
    raise AudioTranscriptionConfigError(
        FIRERED_PYTHON_NOT_FOUND_ERROR.format(
            candidates=", ".join(str(path) for path in candidates),
            environment_variable=FIRERED_PYTHON_ENV,
        )
    )


def load_optional_path_env(name: str, default_relative_path: Path) -> Path:
    value = get_optional_env(name)
    if value is not None:
        return Path(value)
    candidates = _candidate_project_paths(default_relative_path)
    return next((path for path in candidates if path.exists()), candidates[-1])


def load_integer_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise AudioTranscriptionConfigError(
            INVALID_INTEGER_ENV_ERROR.format(name=name, value=value)
        ) from error


def load_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed_value = float(value)
    except ValueError as error:
        raise AudioTranscriptionConfigError(
            INVALID_FLOAT_ENV_ERROR.format(name=name, value=value)
        ) from error
    if not math.isfinite(parsed_value):
        raise AudioTranscriptionConfigError(
            INVALID_FLOAT_ENV_ERROR.format(name=name, value=value)
        )
    return parsed_value


def _candidate_project_paths(relative_path: Path) -> tuple[Path, ...]:
    working_directory_path = Path.cwd() / relative_path
    source_path = SOURCE_PROJECT_ROOT / relative_path
    if source_path == working_directory_path:
        return (source_path,)
    return working_directory_path, source_path
