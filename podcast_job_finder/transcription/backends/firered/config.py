from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


DEFAULT_ORT_PROVIDER: Final = "CPUExecutionProvider"
DEFAULT_ORT_INTRA_OP_THREADS: Final = 4


class FireRedConfigError(ValueError):
    """FireRed 本地转写配置无效。"""


@dataclass(slots=True, frozen=True)
class FireRedProcessConfig:
    python_executable: Path
    ort_provider: str = DEFAULT_ORT_PROVIDER
    ort_intra_op_threads: int = DEFAULT_ORT_INTRA_OP_THREADS

    def __post_init__(self) -> None:
        if not self.python_executable.is_file():
            raise FireRedConfigError(
                f"FIRERED_PYTHON 指向的文件不存在：{self.python_executable}"
            )
        if self.ort_intra_op_threads <= 0:
            raise FireRedConfigError("FIRERED_ORT_INTRA_OP_THREADS 必须大于 0。")
