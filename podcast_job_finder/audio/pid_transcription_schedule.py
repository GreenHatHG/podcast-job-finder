from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Final, Literal, TypeVar

from podcast_job_finder.companies.episode_runner import EpisodeWorkItem


SEQUENTIAL_PROCESSING_MODE: Final = "sequential"
DOWNLOAD_FIRST_PROCESSING_MODE: Final = "download-first"
AudioProcessingMode = Literal["sequential", "download-first"]
SUPPORTED_AUDIO_PROCESSING_MODES: Final = (
    SEQUENTIAL_PROCESSING_MODE,
    DOWNLOAD_FIRST_PROCESSING_MODE,
)
DEFAULT_AUDIO_PROCESSING_MODE: Final = SEQUENTIAL_PROCESSING_MODE

PreparedEpisode = TypeVar("PreparedEpisode")
EpisodeResult = TypeVar("EpisodeResult")

logger = logging.getLogger(__name__)


def run_audio_processing_schedule(
    *,
    work_items: Sequence[EpisodeWorkItem],
    processing_mode: AudioProcessingMode,
    prepare_episode: Callable[[EpisodeWorkItem], PreparedEpisode],
    process_episode: Callable[[PreparedEpisode], EpisodeResult],
) -> list[EpisodeResult]:
    if processing_mode == DOWNLOAD_FIRST_PROCESSING_MODE:
        logger.info("开始预下载节目音频：总数=%d", len(work_items))
        prepared_episodes = [prepare_episode(work_item) for work_item in work_items]
        logger.info("节目音频预下载阶段完成，开始逐集转写")
        return [process_episode(episode) for episode in prepared_episodes]

    return [process_episode(prepare_episode(work_item)) for work_item in work_items]
