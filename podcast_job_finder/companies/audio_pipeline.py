from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from podcast_job_finder.companies._transcript_extraction_scheduler import (
    TranscriptExtractionExecutionResult,
    TranscriptExtractionRequest,
    extract_companies_from_transcripts,
)
from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.pipeline_results import (
    EPISODE_RESULT_INCOMPLETE_ERROR,
    BatchEpisodePipelineResult,
    FailedCompanyEpisodeResult,
    SuccessfulCompanyEpisodeResult,
)
from podcast_job_finder.companies.runtime import EpisodeExtractionRuntime
from podcast_job_finder.companies.transcript_extraction import (
    TranscriptExtractionOutcome,
)
from podcast_job_finder.episode.models import EpisodeResult, EpisodeWorkItem
from podcast_job_finder.transcription.manifest import (
    TranscriptionManifestError,
    load_episode_transcript,
)
from podcast_job_finder.transcription.pipeline_results import (
    BatchAudioTranscriptionResult,
    EpisodeTranscriptionResult,
    SuccessfulEpisodeTranscriptionResult,
)


COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME: Final = "company_extraction"
INVALID_TRANSCRIPTION_PATH_ERROR: Final = "节目结果缺少有效的 transcription_path。"
INVALID_EPISODE_URL_ERROR: Final = "节目结果缺少有效的 episode_url。"

EXPECTED_PREPARATION_ERRORS: Final = (
    TranscriptionManifestError,
    OSError,
    ValueError,
)

type _PreparedExtraction = tuple[
    int,
    SuccessfulEpisodeTranscriptionResult,
    TranscriptExtractionRequest,
]

logger = logging.getLogger(__name__)


def run_batch_audio_company_extraction(
    *,
    # 普通音频模式会传入本次从 RSS 选中的节目，--max-episodes 可以限制节目数量。
    # 转写成功的结果包含 transcription_path，会继续读取转写文本并提取公司；转写
    # 失败的结果只用于在最终报告中保留失败原因，进入本函数后会直接跳过，不会读取
    # 转写文本，也不会发送公司提取的 LLM 请求。
    # --extract-only 模式只传入已经存在 transcription.json 的节目；缺少该文件的
    # 节目在调用本函数前已经被跳过。该对象不包含 RssFeed 对象本身。
    transcription_result: BatchAudioTranscriptionResult,
    runtime: EpisodeExtractionRuntime,
    resume: bool = False,
) -> BatchEpisodePipelineResult:
    episode_results, prepared_extractions = _prepare_extraction_batch(
        transcription_result.episode_results,
        resume=resume,
    )

    if prepared_extractions:
        with ThreadPoolExecutor(
            max_workers=runtime.llm.max_in_flight_requests
        ) as request_executor:
            execution_results = extract_companies_from_transcripts(
                requests=[prepared[2] for prepared in prepared_extractions],
                runtime=runtime,
                request_executor=request_executor,
            )
        for prepared, execution_result in zip(
            prepared_extractions,
            execution_results,
            strict=True,
        ):
            record_index, episode_transcription, request = prepared
            episode_results[record_index] = _build_execution_result(
                episode_transcription,
                work_item=request.work_item,
                execution_result=execution_result,
            )

    if any(result is None for result in episode_results):
        raise RuntimeError(EPISODE_RESULT_INCOMPLETE_ERROR)
    finalized_results: list[EpisodeResult] = [
        result for result in episode_results if result is not None
    ]
    success_count = sum(
        isinstance(result, SuccessfulCompanyEpisodeResult)
        for result in finalized_results
    )
    return BatchEpisodePipelineResult(
        episode_results=finalized_results,
        success_count=success_count,
        fail_count=len(finalized_results) - success_count,
    )


def _prepare_extraction_batch(
    episode_transcriptions: list[EpisodeTranscriptionResult],
    *,
    resume: bool,
) -> tuple[list[EpisodeResult | None], list[_PreparedExtraction]]:
    logger.info(
        "开始准备音频公司提取：节目数=%d",
        len(episode_transcriptions),
    )
    episode_results: list[EpisodeResult | None] = [None] * len(episode_transcriptions)
    prepared_extractions: list[_PreparedExtraction] = []

    for record_index, episode_transcription in enumerate(episode_transcriptions):
        if not isinstance(
            episode_transcription,
            SuccessfulEpisodeTranscriptionResult,
        ):
            episode_results[record_index] = episode_transcription
            continue
        try:
            request = _prepare_extraction_request(
                episode_transcription,
                resume=resume,
            )
        except EXPECTED_PREPARATION_ERRORS as error:
            episode_results[record_index] = _build_error_result(
                episode_transcription,
                error,
            )
            continue
        prepared_extractions.append((record_index, episode_transcription, request))
    logger.info(
        "音频公司提取准备完成：节目数=%d 可提取=%d 失败=%d",
        len(episode_transcriptions),
        len(prepared_extractions),
        sum(result is not None for result in episode_results),
    )
    return episode_results, prepared_extractions


def _prepare_extraction_request(
    episode_transcription: SuccessfulEpisodeTranscriptionResult,
    *,
    resume: bool,
) -> TranscriptExtractionRequest:
    transcription_path = _require_path(episode_transcription.transcription_path)
    work_item = _build_work_item(episode_transcription.episode)
    transcript = load_episode_transcript(transcription_path)
    checkpoint_root = (
        transcription_path.parent / COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME
    ).resolve()
    return TranscriptExtractionRequest(
        work_item=work_item,
        title=transcript.title or work_item.title or "",
        segments=transcript.segments,
        checkpoint_store=LlmCheckpointStore(str(checkpoint_root)),
        resume=resume,
    )


def _build_execution_result(
    episode_transcription: SuccessfulEpisodeTranscriptionResult,
    *,
    work_item: EpisodeWorkItem,
    execution_result: TranscriptExtractionExecutionResult,
) -> EpisodeResult:
    if isinstance(execution_result, Exception):
        return _build_error_result(episode_transcription, execution_result)
    return _build_success_result(
        episode_transcription,
        work_item=work_item,
        outcome=execution_result,
    )


def _build_success_result(
    episode_transcription: SuccessfulEpisodeTranscriptionResult,
    *,
    work_item: EpisodeWorkItem,
    outcome: TranscriptExtractionOutcome,
) -> SuccessfulCompanyEpisodeResult:
    logger.info(
        "音频公司提取完成：eid=%s chunks=%d companies=%d",
        episode_transcription.episode.eid,
        outcome.chunk_count,
        len(outcome.extraction_result.companies),
    )
    return SuccessfulCompanyEpisodeResult(
        episode=work_item,
        extraction_result=outcome.extraction_result,
        transcription_result=episode_transcription,
        extraction_chunk_count=outcome.chunk_count,
        candidate_company_count=outcome.candidate_count,
        extraction_cached=outcome.cached,
    )


def _build_error_result(
    episode_transcription: SuccessfulEpisodeTranscriptionResult,
    error: Exception,
) -> FailedCompanyEpisodeResult:
    logger.info(
        "音频公司提取失败：eid=%s error=%s",
        episode_transcription.episode.eid,
        error,
    )
    return FailedCompanyEpisodeResult(
        episode=episode_transcription.episode,
        error=str(error),
        transcription_result=episode_transcription,
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
