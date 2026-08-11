from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


DEFAULT_ORT_PROVIDER: Final = "CPUExecutionProvider"
DEFAULT_ORT_INTRA_OP_THREADS: Final = 4


@dataclass(slots=True, frozen=True)
class FireRedProcessConfig:
    python_executable: Path
    ort_provider: str = DEFAULT_ORT_PROVIDER
    ort_intra_op_threads: int = DEFAULT_ORT_INTRA_OP_THREADS
