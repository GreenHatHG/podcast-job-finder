"""项目级异常类型，用于标识异常的处理边界。"""

from __future__ import annotations


class PodcastJobFinderError(Exception):
    """程序能够向调用方说明和处理的预期错误。"""


class ConfigurationError(PodcastJobFinderError):
    """运行配置无效，通常会阻止整个命令启动。"""


class EpisodeProcessingError(PodcastJobFinderError):
    """单个节目处理失败，但批量任务可以继续处理其他节目。"""
