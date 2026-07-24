from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from podcast_job_finder.audio.pid_transcription import (
    RESULT_STATUS_SUCCESS,
    PidAudioTranscriptionResult,
)
from podcast_job_finder.audio.transcription_manifest import (
    TranscriptionManifestError,
    load_episode_transcription_manifest,
)
from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.episode_runner import (
    EpisodeExtractionRuntime,
    EpisodeWorkItem,
)
from podcast_job_finder.companies.models import CompanyExtractionError
from podcast_job_finder.companies.pipeline import PidEpisodePipelineResult
from podcast_job_finder.companies.transcript_extraction import (
    extract_companies_from_transcript,
)
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleLlmError,
)


COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME: Final = "company_extraction"
RESULT_STATUS_ERROR: Final = "error"
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


def run_pid_audio_company_extraction(
    *,
    transcription_result: PidAudioTranscriptionResult,
    runtime: EpisodeExtractionRuntime,
) -> PidEpisodePipelineResult:
    episode_results = [
        _extract_episode_record(record, runtime=runtime)
        if record.get("status") == RESULT_STATUS_SUCCESS
        else dict(record)
        for record in transcription_result.episode_results
    ]
    success_count = sum(
        1 for result in episode_results if result.get("status") == RESULT_STATUS_SUCCESS
    )
    return PidEpisodePipelineResult(
        episode_results=episode_results,
        success_count=success_count,
        fail_count=len(episode_results) - success_count,
    )


def _extract_episode_record(
    record: dict[str, object],
    *,
    runtime: EpisodeExtractionRuntime,
) -> dict[str, object]:
    try:
        transcription_path = _require_path(record.get("transcription_path"))
        work_item = _build_work_item(record)
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
        result = dict(record)
        result.update(
            {
                "companies": [
                    company.to_dict() for company in outcome.extraction_result.companies
                ],
                "filtered_count": outcome.extraction_result.filtered_count,
                "extraction_chunk_count": outcome.chunk_count,
                "candidate_company_count": outcome.candidate_count,
                "extraction_cached": outcome.cached,
            }
        )
        logger.info(
            "音频公司提取完成：eid=%s chunks=%d companies=%d",
            record.get("eid"),
            outcome.chunk_count,
            len(outcome.extraction_result.companies),
        )
        return result
    except EXPECTED_EXTRACTION_ERRORS as error:
        logger.info("音频公司提取失败：eid=%s error=%s", record.get("eid"), error)
        result = dict(record)
        result.update({"status": RESULT_STATUS_ERROR, "error": str(error)})
        return result


def _require_path(value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(INVALID_TRANSCRIPTION_PATH_ERROR)
    return Path(value)


def _build_work_item(record: dict[str, object]) -> EpisodeWorkItem:
    episode_url = record.get("episode_url")
    if not isinstance(episode_url, str) or not episode_url.strip():
        raise ValueError(INVALID_EPISODE_URL_ERROR)
    return EpisodeWorkItem(
        episode_url=episode_url,
        eid=_optional_text(record.get("eid")),
        title=_optional_text(record.get("title")),
        pub_date=_optional_text(record.get("pub_date")),
    )


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized_value = value.strip()
    return normalized_value or None
