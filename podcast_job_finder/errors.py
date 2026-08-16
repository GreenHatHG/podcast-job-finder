"""项目预期异常的公共分类；具体捕获边界决定停止还是继续。"""

from __future__ import annotations


class PodcastJobFinderError(Exception):
    """可以直接向用户展示的预期错误；本类型不表示任务能否继续。"""


class ConfigurationError(PodcastJobFinderError):
    """运行配置无效；命令入口捕获后打印错误并停止。"""


class EpisodeProcessingError(PodcastJobFinderError):
    """当前节目处理失败；节目批量边界记录失败后继续其他节目。"""
