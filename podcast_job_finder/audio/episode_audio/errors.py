from podcast_job_finder.errors import EpisodeProcessingError, PodcastJobFinderError


class EpisodeAudioDownloadError(EpisodeProcessingError, RuntimeError):
    """Raised when an episode audio file cannot be downloaded."""


class EpisodeAudioNotFoundError(EpisodeAudioDownloadError):
    """节目音频地址重试后仍返回 HTTP 404。"""


class EpisodeAudioCleanupError(PodcastJobFinderError, RuntimeError):
    """删除节目音频文件失败；命令入口打印错误后停止。"""
