from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from podcast_job_finder.transcription.pipeline_results import (
    BatchAudioTranscriptionResult,
    SuccessfulEpisodeTranscriptionResult,
)
from podcast_job_finder.transcription.manifest import (
    TranscriptionManifestError,
    load_episode_transcription_manifest,
)
from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.runtime import EpisodeExtractionRuntime
from podcast_job_finder.episode.models import EpisodeResult, EpisodeWorkItem
from podcast_job_finder.companies.models import CompanyExtractionError
from podcast_job_finder.companies.pipeline_results import (
    BatchEpisodePipelineResult,
    CompanyEpisodeResult,
    FailedCompanyEpisodeResult,
    SuccessfulCompanyEpisodeResult,
)
from podcast_job_finder.companies.transcript_extraction import (
    extract_companies_from_transcript,
)
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleLlmError,
)


COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME: Final = "company_extraction"
INVALID_TRANSCRIPTION_PATH_ERROR: Final = "节目结果缺少有效的 transcription_path。"
INVALID_EPISODE_URL_ERROR: Final = "节目结果缺少有效的 episode_url。"

EXPECTED_EXTRACTION_ERRORS: Final = (
    CompanyExtractionError,
    EmptyLlmResponseError,
    OpenAiCompatibleLlmError,
    TranscriptionManifestError,
    OSError,
    ValueError,
)

logger = logging.getLogger(__name__)


def run_batch_audio_company_extraction(
    *,
    transcription_result: BatchAudioTranscriptionResult,
    runtime: EpisodeExtractionRuntime,
) -> BatchEpisodePipelineResult:
    episode_results: list[EpisodeResult] = []
    for transcription_episode_result in transcription_result.episode_results:
        if isinstance(
            transcription_episode_result,
            SuccessfulEpisodeTranscriptionResult,
        ):
            episode_results.append(
                _extract_episode_result(
                    transcription_episode_result,
                    runtime=runtime,
                )
            )
        else:
            episode_results.append(transcription_episode_result)
    success_count = sum(
        isinstance(result, SuccessfulCompanyEpisodeResult) for result in episode_results
    )
    return BatchEpisodePipelineResult(
        episode_results=episode_results,
        success_count=success_count,
        fail_count=len(episode_results) - success_count,
    )


def _extract_episode_result(
    transcription_result: SuccessfulEpisodeTranscriptionResult,
    *,
    runtime: EpisodeExtractionRuntime,
) -> CompanyEpisodeResult:
    try:
        transcription_path = _require_path(transcription_result.transcription_path)
        work_item = _build_work_item(transcription_result.episode)
        manifest = load_episode_transcription_manifest(transcription_path)
        outcome = extract_companies_from_transcript(
            work_item=work_item,
            title=manifest.title or work_item.title or "",
            segments=manifest.segments,
            runtime=runtime,
            checkpoint_store=LlmCheckpointStore(
                str(transcription_path.parent / COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME)
            ),
        )
        logger.info(
            "音频公司提取完成：eid=%s chunks=%d companies=%d",
            transcription_result.episode.eid,
            outcome.chunk_count,
            len(outcome.extraction_result.companies),
        )
        return SuccessfulCompanyEpisodeResult(
            episode=work_item,
            extraction_result=outcome.extraction_result,
            transcription_result=transcription_result,
            extraction_chunk_count=outcome.chunk_count,
            candidate_company_count=outcome.candidate_count,
            extraction_cached=outcome.cached,
        )
    except EXPECTED_EXTRACTION_ERRORS as error:
        logger.info(
            "音频公司提取失败：eid=%s error=%s",
            transcription_result.episode.eid,
            error,
        )
        return FailedCompanyEpisodeResult(
            episode=transcription_result.episode,
            error=str(error),
            transcription_result=transcription_result,
        )


def _require_path(value: str) -> Path:
    if not value.strip():
        raise ValueError(INVALID_TRANSCRIPTION_PATH_ERROR)
    return Path(value)


def _build_work_item(episode: EpisodeWorkItem) -> EpisodeWorkItem:
    if not episode.episode_url.strip():
        raise ValueError(INVALID_EPISODE_URL_ERROR)
    return EpisodeWorkItem(
        episode_url=episode.episode_url,
        eid=_optional_text(episode.eid),
        title=_optional_text(episode.title),
        pub_date=_optional_text(episode.pub_date),
    )


def _optional_text(value: str | None) -> str | None:
    normalized_value = (value or "").strip()
    return normalized_value or None
