from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Final, Sequence

from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.episode_runner import (
    CompletedEpisodeExtraction,
    PreparedEpisodeLlmWork,
    restore_or_prepare_episode_work,
    run_prepared_episode_llm_work,
)
from podcast_job_finder.companies.pipeline_results import (
    EPISODE_RESULT_INCOMPLETE_ERROR,
    BatchEpisodePipelineResult,
    CompanyEpisodeResult,
    FailedCompanyEpisodeResult,
    SuccessfulCompanyEpisodeResult,
)
from podcast_job_finder.companies.runtime import EpisodeExtractionRuntime
from podcast_job_finder.episode.models import EpisodeResult, EpisodeWorkItem
from podcast_job_finder.companies.models import CompanyExtractionError
from podcast_job_finder.companies.page_loader import (
    DEFAULT_EPISODE_PAGE_LOADER,
    EpisodePageLoaderProtocol,
    RateLimitedEpisodePageLoader,
)
from podcast_job_finder.llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
)
from podcast_job_finder.llm.rate_limit import format_rate
from podcast_job_finder.episode import EpisodeParseError
from podcast_job_finder.tracing import trace_id_var


TASK_QUEUE_MAX_SIZE: Final = 10
QUEUE_WAIT_TIMEOUT_SECONDS: Final = 0.5
PRODUCER_THREAD_NAME: Final = "episode-prompt-producer"
CONSUMER_THREAD_NAME: Final = "episode-llm-consumer"
QUEUE_SENTINEL: Final = object()
logger = logging.getLogger(__name__)

EXPECTED_EPISODE_ERRORS: Final = (
    CompanyExtractionError,
    EmptyLlmResponseError,
    EpisodeParseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
    ValueError,
)


@dataclass(slots=True, frozen=True)
class _QueuedEpisodeWork:
    episode_index: int
    total_episodes: int
    work_item: EpisodeWorkItem
    prepared_work: PreparedEpisodeLlmWork
    trace_id: str


class _FatalErrorState:
    def __init__(self) -> None:
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def set(self, error: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = error

    def get(self) -> BaseException | None:
        with self._lock:
            return self._error


@dataclass(slots=True)
class _PipelineSharedState:
    checkpoint_store: LlmCheckpointStore
    task_queue: queue.Queue[object]
    episode_results: list[CompanyEpisodeResult | None]
    fatal_error_state: _FatalErrorState


def run_batch_episode_pipeline(
    *,
    work_items: Sequence[EpisodeWorkItem],
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
    page_fetch_rate_per_minute: float | None,
) -> BatchEpisodePipelineResult:
    logger.info(
        "启动节目流水线：总数=%d 单集页面请求速率=%s",
        len(work_items),
        format_rate(page_fetch_rate_per_minute),
    )
    shared_state = _PipelineSharedState(
        checkpoint_store=checkpoint_store,
        task_queue=queue.Queue(maxsize=TASK_QUEUE_MAX_SIZE),
        episode_results=[None] * len(work_items),
        fatal_error_state=_FatalErrorState(),
    )
    _run_pipeline_workers(
        work_items=work_items,
        runtime=runtime,
        page_fetch_rate_per_minute=page_fetch_rate_per_minute,
        shared_state=shared_state,
    )
    return _build_pipeline_result(shared_state)


def _run_pipeline_workers(
    *,
    work_items: Sequence[EpisodeWorkItem],
    runtime: EpisodeExtractionRuntime,
    page_fetch_rate_per_minute: float | None,
    shared_state: _PipelineSharedState,
) -> None:
    page_loader = RateLimitedEpisodePageLoader(
        DEFAULT_EPISODE_PAGE_LOADER,
        page_fetch_rate_per_minute,
    )
    producer_thread = threading.Thread(
        name=PRODUCER_THREAD_NAME,
        target=_produce_episode_tasks,
        args=(work_items, runtime, page_loader, shared_state),
    )
    consumer_thread = threading.Thread(
        name=CONSUMER_THREAD_NAME,
        target=_consume_episode_tasks,
        args=(
            runtime,
            shared_state,
        ),
    )
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()


def _build_pipeline_result(
    shared_state: _PipelineSharedState,
) -> BatchEpisodePipelineResult:
    fatal_error = shared_state.fatal_error_state.get()
    if fatal_error is not None:
        raise fatal_error

    if any(result is None for result in shared_state.episode_results):
        raise RuntimeError(EPISODE_RESULT_INCOMPLETE_ERROR)

    finalized_results: list[EpisodeResult] = [
        result for result in shared_state.episode_results if result is not None
    ]
    success_count = sum(
        isinstance(result, SuccessfulCompanyEpisodeResult)
        for result in finalized_results
    )
    fail_count = len(finalized_results) - success_count
    return BatchEpisodePipelineResult(
        episode_results=finalized_results,
        success_count=success_count,
        fail_count=fail_count,
    )


def _produce_episode_tasks(
    work_items: Sequence[EpisodeWorkItem],
    runtime: EpisodeExtractionRuntime,
    page_loader: EpisodePageLoaderProtocol,
    shared_state: _PipelineSharedState,
) -> None:
    total_episodes = len(work_items)
    try:
        for episode_index, work_item in enumerate(work_items):
            if shared_state.fatal_error_state.get() is not None:
                return

            trace_id = _build_trace_id(episode_index, work_item)
            trace_id_var.set(trace_id)

            logger.info(
                "生产节目任务 %d/%d：%s",
                episode_index + 1,
                total_episodes,
                work_item.title or work_item.episode_url,
            )
            try:
                episode_work = restore_or_prepare_episode_work(
                    work_item=work_item,
                    runtime=runtime,
                    checkpoint_store=shared_state.checkpoint_store,
                    page_loader=page_loader,
                )
                if isinstance(episode_work, CompletedEpisodeExtraction):
                    shared_state.episode_results[episode_index] = (
                        SuccessfulCompanyEpisodeResult(
                            episode=episode_work.episode,
                            extraction_result=episode_work.extraction_result,
                        )
                    )
                    continue

                _put_queue_item(
                    task_queue=shared_state.task_queue,
                    payload=_QueuedEpisodeWork(
                        episode_index=episode_index,
                        total_episodes=total_episodes,
                        work_item=work_item,
                        prepared_work=episode_work,
                        trace_id=trace_id,
                    ),
                    fatal_error_state=shared_state.fatal_error_state,
                )
            except EXPECTED_EPISODE_ERRORS as error:
                logger.info("节目生产失败：%s", error)
                shared_state.episode_results[episode_index] = (
                    FailedCompanyEpisodeResult(
                        episode=work_item,
                        error=str(error),
                    )
                )
            finally:
                trace_id_var.set("-")

        _put_queue_item(
            task_queue=shared_state.task_queue,
            payload=QUEUE_SENTINEL,
            fatal_error_state=shared_state.fatal_error_state,
        )
    except Exception as error:  # pylint: disable=broad-exception-caught
        _handle_pipeline_error(shared_state, error)


def _consume_episode_tasks(
    runtime: EpisodeExtractionRuntime,
    shared_state: _PipelineSharedState,
) -> None:
    try:
        while True:
            queued_work = _get_queue_item(
                task_queue=shared_state.task_queue,
                fatal_error_state=shared_state.fatal_error_state,
            )
            if queued_work is None:
                return

            trace_id_var.set(queued_work.trace_id)

            logger.info(
                "消费节目任务 %d/%d：%s",
                queued_work.episode_index + 1,
                queued_work.total_episodes,
                queued_work.work_item.title or queued_work.work_item.episode_url,
            )
            try:
                completed_extraction = run_prepared_episode_llm_work(
                    prepared_work=queued_work.prepared_work,
                    runtime=runtime,
                    checkpoint_store=shared_state.checkpoint_store,
                )
                logger.info(
                    "节目处理完成：提取到 %d 家公司，过滤 %d 家",
                    len(completed_extraction.extraction_result.companies),
                    completed_extraction.extraction_result.filtered_count,
                )
                shared_state.episode_results[queued_work.episode_index] = (
                    SuccessfulCompanyEpisodeResult(
                        episode=completed_extraction.episode,
                        extraction_result=completed_extraction.extraction_result,
                    )
                )
            except EXPECTED_EPISODE_ERRORS as error:
                logger.info("节目消费失败：%s", error)
                shared_state.episode_results[queued_work.episode_index] = (
                    FailedCompanyEpisodeResult(
                        episode=queued_work.work_item,
                        error=str(error),
                    )
                )
            finally:
                trace_id_var.set("-")

    except Exception as error:  # pylint: disable=broad-exception-caught
        _handle_pipeline_error(shared_state, error)


def _build_trace_id(episode_index: int, work_item: EpisodeWorkItem) -> str:
    """根据序号和 eid 生成有意义的 trace_id，例如 001-5f4a8b2c"""
    eid = work_item.eid or work_item.episode_url.rstrip("/").split("/")[-1]
    eid_short = eid[-8:] if len(eid) >= 8 else eid.ljust(8, "0")
    return f"{episode_index:03d}-{eid_short}"


def _put_queue_item(
    *,
    task_queue: queue.Queue[object],
    payload: object,
    fatal_error_state: _FatalErrorState,
) -> None:
    while True:
        if fatal_error_state.get() is not None:
            return
        try:
            task_queue.put(payload, timeout=QUEUE_WAIT_TIMEOUT_SECONDS)
            return
        except queue.Full:
            continue


def _handle_pipeline_error(
    shared_state: _PipelineSharedState,
    error: Exception,
) -> None:
    shared_state.fatal_error_state.set(error)


def _get_queue_item(
    *,
    task_queue: queue.Queue[object],
    fatal_error_state: _FatalErrorState,
) -> _QueuedEpisodeWork | None:
    while True:
        if fatal_error_state.get() is not None:
            return None
        try:
            payload = task_queue.get(timeout=QUEUE_WAIT_TIMEOUT_SECONDS)
        except queue.Empty:
            continue
        if payload is QUEUE_SENTINEL:
            return None
        if not isinstance(payload, _QueuedEpisodeWork):
            raise TypeError("节目流水线收到未知队列任务。")
        return payload
