"""分布式追踪支持：提供上下文 trace_id 及日志过滤器。"""

from __future__ import annotations

import contextvars
import logging

# 全局上下文变量，跨线程隔离，默认值为 "-" 表示未设置
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)


class TraceIdFormatter(logging.Formatter):
    """在格式化每条日志时，从上下文中注入 trace_id。"""

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = trace_id_var.get()
        return super().format(record)
