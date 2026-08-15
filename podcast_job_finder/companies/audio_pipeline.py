from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

from podcast_job_finder.transcription.pipeline_results import (
    RESULT_STATUS_SUCCESS,
    BatchAudioTranscriptionResult,
)
from podcast_job_finder.transcription.manifest import (
    TranscriptionManifestError,
    load_episode_transcription_manifest,
)
from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.runtime import EpisodeExtractionRuntime
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.companies.pipeline_results import (
    EPISODE_RESULT_INCOMPLETE_ERROR,
    BatchEpisodePipelineResult,
)
from podcast_job_finder.companies._transcript_extraction_scheduler import (
    TranscriptExtractionExecutionResult,
    TranscriptExtractionRequest,
    extract_companies_from_transcripts,
)
from podcast_job_finder.companies.transcript_extraction import (
    TranscriptExtractionOutcome,
)


COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME: Final = "company_extraction"
RESULT_STATUS_ERROR: Final = "error"
INVALID_TRANSCRIPTION_PATH_ERROR: Final = "节目结果缺少有效的 transcription_path。"
INVALID_EPISODE_URL_ERROR: Final = "节目结果缺少有效的 episode_url。"

EXPECTED_PREPARATION_ERRORS: Final = (
    TranscriptionManifestError,
    OSError,
    ValueError,
)

logger = logging.getLogger(__name__)


def run_batch_audio_company_extraction(
    *,
    # 普通音频模式会传入本次从 RSS 选中的节目，--max-episodes 可以限制节目数量。
    # 转写成功的记录包含 transcription_path，会继续读取转写文本并提取公司；转写
    # 失败的记录只用于在最终报告中保留失败原因，进入本函数后会直接跳过，不会读取
    # 转写文本，也不会发送公司提取的 LLM 请求。
    # --extract-only 模式只传入已经存在 transcription.json 的节目；缺少该文件的
    # 节目在调用本函数前已经被跳过。该对象不包含 RssFeed 对象本身。
    transcription_result: BatchAudioTranscriptionResult,
    runtime: EpisodeExtractionRuntime,
) -> BatchEpisodePipelineResult:
    records = transcription_result.episode_results
    episode_results, requests, request_record_indexes = _prepare_extraction_batch(
        records
    )

    if requests:
        with ThreadPoolExecutor(
            max_workers=runtime.llm.max_in_flight_requests
        ) as request_executor:
            execution_results = extract_companies_from_transcripts(
                requests=requests,
                runtime=runtime,
                request_executor=request_executor,
            )
        for record_index, execution_result in zip(
            request_record_indexes,
            execution_results,
            strict=True,
        ):
            record = records[record_index]
            episode_results[record_index] = _build_execution_record(
                record,
                execution_result,
            )

    if any(result is None for result in episode_results):
        raise RuntimeError(EPISODE_RESULT_INCOMPLETE_ERROR)
    finalized_results = [result for result in episode_results if result is not None]
    success_count = sum(
        1
        for result in finalized_results
        if result.get("status") == RESULT_STATUS_SUCCESS
    )
    return BatchEpisodePipelineResult(
        episode_results=finalized_results,
        success_count=success_count,
        fail_count=len(finalized_results) - success_count,
    )


def _prepare_extraction_batch(
    records: list[dict[str, object]],
) -> tuple[
    list[dict[str, object] | None],
    list[TranscriptExtractionRequest],
    list[int],
]:
    episode_results: list[dict[str, object] | None] = [None] * len(records)
    requests: list[TranscriptExtractionRequest] = []
    request_record_indexes: list[int] = []

    for record_index, record in enumerate(records):
        if record.get("status") != RESULT_STATUS_SUCCESS:
            episode_results[record_index] = dict(record)
            continue
        try:
            request = _prepare_extraction_request(record)
        except EXPECTED_PREPARATION_ERRORS as error:
            episode_results[record_index] = _build_error_record(record, error)
            continue
        requests.append(request)
        request_record_indexes.append(record_index)
    return episode_results, requests, request_record_indexes


def _prepare_extraction_request(
    record: dict[str, object],
) -> TranscriptExtractionRequest:
    transcription_path = _require_path(record.get("transcription_path"))
    work_item = _build_work_item(record)
    manifest = load_episode_transcription_manifest(transcription_path)
    checkpoint_root = (
        transcription_path.parent / COMPANY_EXTRACTION_CHECKPOINT_DIR_NAME
    ).resolve()
    return TranscriptExtractionRequest(
        work_item=work_item,
        title=manifest.title or work_item.title or "",
        segments=manifest.segments,
        checkpoint_store=LlmCheckpointStore(str(checkpoint_root)),
    )


def _build_execution_record(
    record: dict[str, object],
    execution_result: TranscriptExtractionExecutionResult,
) -> dict[str, object]:
    if isinstance(execution_result, Exception):
        return _build_error_record(record, execution_result)
    return _build_success_record(record, execution_result)


def _build_success_record(
    record: dict[str, object],
    outcome: TranscriptExtractionOutcome,
) -> dict[str, object]:
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


def _build_error_record(
    record: dict[str, object],
    error: Exception,
) -> dict[str, object]:
    logger.info("音频公司提取失败：eid=%s error=%s", record.get("eid"), error)
    result = dict(record)
    result.update(
        {
            "status": RESULT_STATUS_ERROR,
            "error": str(error),
        }
    )
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
