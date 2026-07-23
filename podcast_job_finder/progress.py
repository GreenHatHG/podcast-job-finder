from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final


# 每跨过 10% 输出一次进度日志。
DEFAULT_PROGRESS_REPORT_INTERVAL_PERCENT: Final = 10
INVALID_TOTAL_ITEMS_ERROR: Final = "total_items 必须大于 0。"


@dataclass(slots=True)
class PercentageProgressLogger:
    """按固定百分比间隔输出串行任务的 INFO 进度日志。

    调用方为每个任务创建一个实例，并在完成一项工作后调用 ``update``。
    实例只保存下一个报告百分比。总量无法被十整除时，日志会显示首次跨过
    阈值时的实际百分比，例如 10.4%。
    """

    # 接收进度日志的 logger。repr 中省略该对象，保持实例输出简洁。
    logger: logging.Logger = field(repr=False)

    # 日志中的任务名称，例如“逐帧语音检测”。
    operation_name: str

    # 创建实例时确定的总工作量，用于计算实际完成百分比。
    total_items: int

    # 下一个需要输出日志的百分比，初始为 10，输出后推进到下一个 10% 倍数。
    _next_report_percent: int = field(
        init=False,
        default=DEFAULT_PROGRESS_REPORT_INTERVAL_PERCENT,
        repr=False,
    )

    def __post_init__(self) -> None:
        """校验计算百分比所需的总工作量。"""
        if self.total_items <= 0:
            raise ValueError(INVALID_TOTAL_ITEMS_ERROR)

    def update(self, completed_items: int) -> None:
        """计算当前百分比，跨过下一个报告阈值时输出日志。"""
        progress_percent = completed_items / self.total_items * 100
        if progress_percent < self._next_report_percent:
            return

        self.logger.info(
            "%s进度：completed=%d total=%d progress=%.1f%%",
            self.operation_name,
            completed_items,
            self.total_items,
            progress_percent,
        )

        # 根据当前实际百分比直接计算下一个 10% 倍数。即使一次 update 跨过
        # 多个阈值，下一次调用也只会等待新的阈值，避免重复报告旧进度。
        self._next_report_percent = (
            int(progress_percent) // DEFAULT_PROGRESS_REPORT_INTERVAL_PERCENT + 1
        ) * DEFAULT_PROGRESS_REPORT_INTERVAL_PERCENT
