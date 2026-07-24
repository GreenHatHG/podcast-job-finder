from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Final, Sequence

from podcast_job_finder.audio.transcription import TranscribedSpeechSegment
from podcast_job_finder.companies.checkpoint import (
    STATUS_SUCCESS,
    LlmCheckpointSavePayload,
    LlmCheckpointStore,
)
from podcast_job_finder.companies.candidate_merge import (
    build_candidate_merge_prompt,
    validate_merged_result,
)
from podcast_job_finder.companies.episode_runner import (
    EpisodeExtractionRuntime,
    EpisodeWorkItem,
)
from podcast_job_finder.companies.extraction import (
    build_company_extraction_input,
    build_company_extraction_prompt,
    normalize_company_mentions,
    run_company_extraction_from_prompt,
)
from podcast_job_finder.companies.models import (
    CompanyExtractionError,
    CompanyExtractionResult,
    CompanyMention,
)
from podcast_job_finder.companies.transcript_chunks import (
    TranscriptChunk,
    build_transcript_chunks,
)
from podcast_job_finder.runtime_signature import build_runtime_signature_hash
from podcast_job_finder.xiaoyuzhou.models import EpisodeInfo


CHUNK_CHECKPOINT_KEY_TEMPLATE: Final = "chunk_{index:04d}"
MERGE_CHECKPOINT_KEY: Final = "merge"
INCOMPLETE_EXTRACTION_ERROR: Final = "LLM 成功结果缺少必要字段。"

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class TranscriptExtractionOutcome:
    extraction_result: CompanyExtractionResult
    chunk_count: int
    candidate_count: int
    cached: bool


@dataclass(slots=True, frozen=True)
class _ExtractionExecutionContext:
    work_item: EpisodeWorkItem
    runtime: EpisodeExtractionRuntime
    checkpoint_store: LlmCheckpointStore


def extract_companies_from_transcript(
    *,
    work_item: EpisodeWorkItem,
    title: str,
    segments: Sequence[TranscribedSpeechSegment],
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
) -> TranscriptExtractionOutcome:
    chunks = build_transcript_chunks(segments)
    if not chunks:
        return TranscriptExtractionOutcome(
            extraction_result=CompanyExtractionResult(),
            chunk_count=0,
            candidate_count=0,
            cached=False,
        )

    context = _ExtractionExecutionContext(
        work_item=work_item,
        runtime=runtime,
        checkpoint_store=checkpoint_store,
    )
    chunk_results: list[CompanyExtractionResult] = []
    all_cached = True
    for chunk in chunks:
        result, cached = _extract_transcript_chunk(
            chunk=chunk,
            title=title,
            context=context,
        )
        chunk_results.append(result)
        all_cached = all_cached and cached

    candidates = [
        company for chunk_result in chunk_results for company in chunk_result.companies
    ]
    if len(chunks) == 1 or not candidates:
        extraction_result = normalize_company_mentions(
            candidates,
            company_blacklist=runtime.company_blacklist,
        )
    else:
        extraction_result, merge_cached = _merge_company_candidates(
            candidates=candidates,
            context=context,
        )
        all_cached = all_cached and merge_cached

    return TranscriptExtractionOutcome(
        extraction_result=extraction_result,
        chunk_count=len(chunks),
        candidate_count=len(candidates),
        cached=all_cached,
    )


def _extract_transcript_chunk(
    *,
    chunk: TranscriptChunk,
    title: str,
    context: _ExtractionExecutionContext,
) -> tuple[CompanyExtractionResult, bool]:
    episode = EpisodeInfo(title=title, content=chunk.text)
    prompt = build_company_extraction_prompt(build_company_extraction_input(episode))
    return _run_cached_extraction(
        checkpoint_key=CHUNK_CHECKPOINT_KEY_TEMPLATE.format(index=chunk.index),
        prompt=prompt,
        context=context,
        company_blacklist=(),
    )


def _merge_company_candidates(
    *,
    candidates: Sequence[CompanyMention],
    context: _ExtractionExecutionContext,
) -> tuple[CompanyExtractionResult, bool]:
    prompt, merge_candidates = build_candidate_merge_prompt(candidates)
    return _run_cached_extraction(
        checkpoint_key=MERGE_CHECKPOINT_KEY,
        prompt=prompt,
        context=context,
        company_blacklist=context.runtime.company_blacklist,
        result_validator=lambda result: validate_merged_result(
            result,
            merge_candidates,
        ),
    )


def _run_cached_extraction(
    *,
    checkpoint_key: str,
    prompt: str,
    context: _ExtractionExecutionContext,
    company_blacklist: Sequence[str],
    result_validator: Callable[[CompanyExtractionResult], None] | None = None,
) -> tuple[CompanyExtractionResult, bool]:
    runtime = context.runtime
    runtime_signature = _build_prompt_runtime_signature(
        runtime,
        prompt,
        company_blacklist=company_blacklist,
    )
    cached_result = _load_cached_result(
        checkpoint_key=checkpoint_key,
        runtime_signature=runtime_signature,
        checkpoint_store=context.checkpoint_store,
        result_validator=result_validator,
    )
    if cached_result is not None:
        logger.info("命中音频公司提取检查点：key=%s", checkpoint_key)
        return cached_result, True

    payload = LlmCheckpointSavePayload(
        episode_key=checkpoint_key,
        episode_url=context.work_item.episode_url,
        title=context.work_item.title,
        pub_date=context.work_item.pub_date,
        runtime_signature=runtime_signature,
        prompt_text=prompt,
    )
    context.checkpoint_store.save_prepared(payload)
    attempt = run_company_extraction_from_prompt(
        prompt,
        runtime.llm_client,
        company_blacklist=company_blacklist,
        retry_config=runtime.retry_config,
        result_validator=result_validator,
    )
    if attempt.error is not None:
        context.checkpoint_store.save_failed(
            payload,
            error_message=str(attempt.error),
            response_text=attempt.response_text,
        )
        raise attempt.error
    if attempt.extraction_result is None or attempt.response_text is None:
        raise ValueError(INCOMPLETE_EXTRACTION_ERROR)

    context.checkpoint_store.save_success(
        payload,
        response_text=attempt.response_text,
        extraction_result=attempt.extraction_result,
    )
    return attempt.extraction_result, False


def _load_cached_result(
    *,
    checkpoint_key: str,
    runtime_signature: str,
    checkpoint_store: LlmCheckpointStore,
    result_validator: Callable[[CompanyExtractionResult], None] | None,
) -> CompanyExtractionResult | None:
    checkpoint = checkpoint_store.load(checkpoint_key)
    if checkpoint is None:
        return None
    if checkpoint.state.runtime_signature != runtime_signature:
        return None
    if checkpoint.state.status != STATUS_SUCCESS:
        return None
    result = CompanyExtractionResult.from_dict(
        {
            "companies": checkpoint.state.companies,
            "filtered_count": checkpoint.state.filtered_count,
        }
    )
    if result_validator is None:
        return result
    try:
        result_validator(result)
    except CompanyExtractionError as error:
        logger.info(
            "音频公司提取检查点未通过结果校验，将重新执行：key=%s error=%s",
            checkpoint_key,
            error,
        )
        return None
    return result


def _build_prompt_runtime_signature(
    runtime: EpisodeExtractionRuntime,
    prompt: str,
    *,
    company_blacklist: Sequence[str],
) -> str:
    return build_runtime_signature_hash(
        {
            "model": runtime.model,
            "base_url": runtime.base_url,
            "api_style": runtime.api_style,
            "company_blacklist": sorted(
                {
                    company_name.strip().casefold()
                    for company_name in company_blacklist
                    if company_name.strip()
                }
            ),
            "prompt": prompt,
        }
    )
