from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Final, Sequence

from podcast_job_finder.companies.checkpoint import (
    STATUS_SUCCESS,
    LlmCheckpoint,
    LlmCheckpointSavePayload,
    LlmCheckpointStore,
)
from podcast_job_finder.companies.candidate_merge import (
    build_candidate_merge_prompt,
    validate_merged_result,
)
from podcast_job_finder.companies.evidence_validation import (
    validate_company_evidence,
)
from podcast_job_finder.companies.runtime import EpisodeExtractionRuntime
from podcast_job_finder.companies.extraction import (
    build_company_extraction_prompt,
    run_company_extraction_from_prompt,
)
from podcast_job_finder.companies.models import (
    CompanyExtractionError,
    CompanyExtractionResult,
    CompanyMention,
)
from podcast_job_finder.companies.transcript_chunks import (
    TranscriptChunk,
)
from podcast_job_finder.episode import EpisodeInfo
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.tracing import trace_id_var
from podcast_job_finder.transcription.models import TranscribedSpeechSegment


CHUNK_CHECKPOINT_KEY_TEMPLATE: Final = "chunk_{index:04d}"
MERGE_CHECKPOINT_KEY: Final = "merge"
INCOMPLETE_EXTRACTION_ERROR: Final = "LLM 成功结果缺少必要字段。"
TRANSCRIPT_CHUNK_EVIDENCE_PROMPT_SUFFIX: Final = """以上待处理文本还需遵守以下规则：
1. evidence 可以截取一段连续原文并省略前后内容，但不要改写或拼接。
2. evidence 必须直接包含对应的 name；空格和换行可以按输出需要调整。
"""

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
    resume: bool


# pylint: disable-next=too-many-arguments
def extract_companies_from_transcript(  # noqa
    *,
    work_item: EpisodeWorkItem,
    title: str,
    segments: Sequence[TranscribedSpeechSegment],
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
    resume: bool = False,
) -> TranscriptExtractionOutcome:
    """提取单个节目的公司，同时保留原有调用接口。"""
    # 延迟导入可避免调度器反向导入本模块中的文本块处理函数时形成循环导入。
    # pylint: disable=import-outside-toplevel
    from concurrent.futures import ThreadPoolExecutor

    from podcast_job_finder.companies._transcript_extraction_scheduler import (
        TranscriptExtractionFailure,
        TranscriptExtractionRequest,
        extract_companies_from_transcripts,
    )
    # pylint: enable=import-outside-toplevel

    with ThreadPoolExecutor(
        max_workers=runtime.llm.max_in_flight_requests
    ) as request_executor:
        execution_result = extract_companies_from_transcripts(
            requests=(
                TranscriptExtractionRequest(
                    work_item=work_item,
                    title=title,
                    segments=tuple(segments),
                    checkpoint_store=checkpoint_store,
                    resume=resume,
                ),
            ),
            runtime=runtime,
            request_executor=request_executor,
        )[0]
    if isinstance(execution_result, TranscriptExtractionFailure):
        raise execution_result.error
    return execution_result


def _extract_transcript_chunk(
    *,
    chunk: TranscriptChunk,
    title: str,
    context: _ExtractionExecutionContext,
) -> tuple[CompanyExtractionResult, bool]:
    episode = EpisodeInfo(title=title, content=chunk.text)
    prompt = "\n".join(
        (
            build_company_extraction_prompt(episode),
            TRANSCRIPT_CHUNK_EVIDENCE_PROMPT_SUFFIX,
        )
    )
    return _run_cached_extraction(
        checkpoint_key=CHUNK_CHECKPOINT_KEY_TEMPLATE.format(index=chunk.index),
        prompt=prompt,
        context=context,
        company_blacklist=(),
        result_validator=lambda result: _validate_transcript_chunk_result(
            result,
            title=title,
            chunk_text=chunk.text,
        ),
    )


def _validate_transcript_chunk_result(
    result: CompanyExtractionResult,
    *,
    title: str,
    chunk_text: str,
) -> None:
    allowed_sources = (title, chunk_text)
    for company in result.companies:
        validate_company_evidence(company, allowed_sources)


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
    episode_id = context.work_item.resolve_episode_id()
    cached_result = None
    if context.resume:
        cached_result = _load_cached_result(
            checkpoint_key=checkpoint_key,
            episode_id=episode_id,
            episode_url=context.work_item.episode_url,
            checkpoint_store=context.checkpoint_store,
            result_validator=result_validator,
        )
    if cached_result is not None:
        logger.info(
            "命中音频公司提取检查点：eid=%s key=%s",
            episode_id,
            checkpoint_key,
        )
        return cached_result, True

    payload = LlmCheckpointSavePayload(
        episode_key=checkpoint_key,
        episode_url=context.work_item.episode_url,
        title=context.work_item.title,
        pub_date=context.work_item.pub_date,
        runtime_signature=None,
        prompt_text=prompt,
    )
    context.checkpoint_store.save_prepared(payload)
    request_trace_id = (
        f"company-extraction/{episode_id or context.work_item.episode_url}/"
        f"{checkpoint_key}"
    )
    trace_id_token = trace_id_var.set(request_trace_id)
    try:
        attempt = run_company_extraction_from_prompt(
            prompt,
            runtime.llm,
            company_blacklist=company_blacklist,
            result_validator=result_validator,
        )
    finally:
        trace_id_var.reset(trace_id_token)
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
    episode_id: str | None,
    episode_url: str,
    checkpoint_store: LlmCheckpointStore,
    result_validator: Callable[[CompanyExtractionResult], None] | None,
) -> CompanyExtractionResult | None:
    checkpoint = checkpoint_store.load(checkpoint_key)
    if checkpoint is None:
        return None
    invalid_reason = _find_invalid_checkpoint_reason(
        checkpoint,
        expected_episode_url=episode_url,
    )
    if invalid_reason is not None:
        logger.info(
            "音频公司提取检查点不可用，将重新执行：eid=%s key=%s reason=%s",
            episode_id,
            checkpoint_key,
            invalid_reason,
        )
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
            "音频公司提取检查点未通过结果校验，将重新执行：eid=%s key=%s error=%s",
            episode_id,
            checkpoint_key,
            error,
        )
        return None
    return result


def _find_invalid_checkpoint_reason(
    checkpoint: LlmCheckpoint,
    *,
    expected_episode_url: str,
) -> str | None:
    if checkpoint.state.status != STATUS_SUCCESS:
        return "状态不是 success"
    if not checkpoint.prompt_text or not checkpoint.prompt_text.strip():
        return "缺少非空的 llm_prompt.txt"
    if not checkpoint.response_text or not checkpoint.response_text.strip():
        return "缺少非空的 llm_response.txt"
    if checkpoint.state.episode_url != expected_episode_url:
        return "episode_url 已变化"
    return None
