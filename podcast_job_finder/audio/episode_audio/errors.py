from podcast_job_finder.errors import EpisodeProcessingError


class EpisodeAudioDownloadError(EpisodeProcessingError, RuntimeError):
    """Raised when an episode audio file cannot be downloaded."""
