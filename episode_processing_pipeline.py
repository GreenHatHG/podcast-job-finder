from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Final, Sequence

from tracing import trace_id_var
from company_extraction import CompanyExtractionError, LlmClientProtocol
from episode_company_runner import (
    EpisodeExtractionOutcome,
    EpisodeExtractionRuntime,
    EpisodeWorkItem,
    PreparedEpisodeLlmWork,
    restore_or_prepare_episode_work,
    run_prepared_episode_llm_work,
)
from extract_xiaoyuzhou_episode import EpisodeParseError
from llm_checkpoint_store import LlmCheckpointStore
from openai_compatible_llm import (
    EmptyLlmResponseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
)


LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE_ENV: Final = (
    "LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE"
)
LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE_ENV: Final = (
    "LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE"
)
INVALID_RATE_ENV_TEMPLATE: Final = "环境变量 {env_name} 必须是大于 0 的数字。"
RESULT_STATUS_SUCCESS: Final = "success"
RESULT_STATUS_ERROR: Final = "error"
TASK_QUEUE_MAX_SIZE: Final = 10
QUEUE_WAIT_TIMEOUT_SECONDS: Final = 0.5
PRODUCER_THREAD_NAME: Final = "episode-prompt-producer"
CONSUMER_THREAD_NAME: Final = "episode-llm-consumer"
QUEUE_SENTINEL: Final = object()
EPISODE_RESULT_INCOMPLETE_ERROR: Final = "节目流水线未生成完整结果。"

logger = logging.getLogger(__name__)

EXPECTED_EPISODE_ERRORS = (
    CompanyExtractionError,
    EmptyLlmResponseError,
    EpisodeParseError,
    OpenAiCompatibleConfigError,
    OpenAiCompatibleLlmError,
    ValueError,
)


class PipelineRateConfigError(ValueError):
    """Raised when the pipeline rate configuration is invalid."""


@dataclass(slots=True, frozen=True)
class PipelineRateConfig:
    producer_rate_per_minute: float | None = None
    consumer_rate_per_minute: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_rate(
            self.producer_rate_per_minute,
            LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE_ENV,
        )
        _validate_optional_rate(
            self.consumer_rate_per_minute,
            LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE_ENV,
        )


@dataclass(slots=True, frozen=True)
class PidEpisodePipelineResult:
    episode_results: list[dict]
    success_count: int
    fail_count: int


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
    episode_results: list[dict | None]
    fatal_error_state: _FatalErrorState


class _PerMinuteRateLimiter:
    def __init__(self, rate_per_minute: float | None) -> None:
        self._min_interval_seconds = (
            None if rate_per_minute is None else 60.0 / rate_per_minute
        )
        self._next_allowed_at: float | None = None

    def wait_turn(self) -> None:
        if self._min_interval_seconds is None:
            return

        current_time = time.monotonic()
        if self._next_allowed_at is None:
            self._next_allowed_at = current_time + self._min_interval_seconds
            return

        sleep_seconds = self._next_allowed_at - current_time
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
            current_time = self._next_allowed_at
        else:
            current_time = time.monotonic()

        self._next_allowed_at = current_time + self._min_interval_seconds


class RateLimitedLlmClient:
    def __init__(
        self,
        wrapped_client: LlmClientProtocol,
        rate_limiter: _PerMinuteRateLimiter,
    ) -> None:
        self._wrapped_client = wrapped_client
        self._rate_limiter = rate_limiter

    def generate(self, prompt: str) -> str:
        self._rate_limiter.wait_turn()
        return self._wrapped_client.generate(prompt)


def load_pipeline_rate_config_from_env() -> PipelineRateConfig:
    return PipelineRateConfig(
        producer_rate_per_minute=_get_optional_rate_env(
            LLM_PIPELINE_PRODUCER_RATE_PER_MINUTE_ENV
        ),
        consumer_rate_per_minute=_get_optional_rate_env(
            LLM_PIPELINE_CONSUMER_RATE_PER_MINUTE_ENV
        ),
    )


def run_pid_episode_pipeline(
    *,
    work_items: Sequence[EpisodeWorkItem],
    runtime: EpisodeExtractionRuntime,
    checkpoint_store: LlmCheckpointStore,
    rate_config: PipelineRateConfig,
) -> PidEpisodePipelineResult:
    logger.info(
        "启动节目流水线：总数=%d 生产速率=%s 消费速率=%s",
        len(work_items),
        _format_rate(rate_config.producer_rate_per_minute),
        _format_rate(rate_config.consumer_rate_per_minute),
    )
    episode_results: list[dict | None] = [None] * len(work_items)
    task_queue: queue.Queue[object] = queue.Queue(maxsize=TASK_QUEUE_MAX_SIZE)
    fatal_error_state = _FatalErrorState()
    shared_state = _PipelineSharedState(
        checkpoint_store=checkpoint_store,
        task_queue=task_queue,
        episode_results=episode_results,
        fatal_error_state=fatal_error_state,
    )
    producer_limiter = _PerMinuteRateLimiter(rate_config.producer_rate_per_minute)
    consumer_runtime = replace(
        runtime,
        llm_client=RateLimitedLlmClient(
            runtime.llm_client,
            _PerMinuteRateLimiter(rate_config.consumer_rate_per_minute),
        ),
    )
    producer_thread = threading.Thread(
        name=PRODUCER_THREAD_NAME,
        target=_produce_episode_tasks,
        args=(work_items, runtime, producer_limiter, shared_state),
    )
    consumer_thread = threading.Thread(
        name=CONSUMER_THREAD_NAME,
        target=_consume_episode_tasks,
        args=(
            consumer_runtime,
            shared_state,
        ),
    )
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()

    fatal_error = fatal_error_state.get()
    if fatal_error is not None:
        raise fatal_error

    if any(result is None for result in episode_results):
        raise RuntimeError(EPISODE_RESULT_INCOMPLETE_ERROR)

    finalized_results = [result for result in episode_results if result is not None]
    success_count = sum(
        1
        for result in finalized_results
        if result.get("status") == RESULT_STATUS_SUCCESS
    )
    fail_count = len(finalized_results) - success_count
    return PidEpisodePipelineResult(
        episode_results=finalized_results,
        success_count=success_count,
        fail_count=fail_count,
    )


def _produce_episode_tasks(
    work_items: Sequence[EpisodeWorkItem],
    runtime: EpisodeExtractionRuntime,
    producer_limiter: _PerMinuteRateLimiter,
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
                )
                if isinstance(episode_work, EpisodeExtractionOutcome):
                    shared_state.episode_results[episode_index] = (
                        _build_success_result_record(episode_work)
                    )
                    continue

                producer_limiter.wait_turn()
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
                    _build_error_result_record(
                        work_item=work_item,
                        error_message=str(error),
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
                extraction_outcome = run_prepared_episode_llm_work(
                    prepared_work=queued_work.prepared_work,
                    runtime=runtime,
                    checkpoint_store=shared_state.checkpoint_store,
                )
                logger.info(
                    "节目处理完成：提取到 %d 家公司，过滤 %d 家",
                    len(extraction_outcome.extraction_result.companies),
                    extraction_outcome.extraction_result.filtered_count,
                )
                shared_state.episode_results[queued_work.episode_index] = (
                    _build_success_result_record(extraction_outcome)
                )
            except EXPECTED_EPISODE_ERRORS as error:
                logger.info("节目消费失败：%s", error)
                shared_state.episode_results[queued_work.episode_index] = (
                    _build_error_result_record(
                        work_item=queued_work.work_item,
                        error_message=str(error),
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


def _build_error_result_record(
    *,
    work_item: EpisodeWorkItem,
    error_message: str,
) -> dict:
    return {
        "status": RESULT_STATUS_ERROR,
        "eid": work_item.eid,
        "title": work_item.title,
        "pub_date": work_item.pub_date,
        "episode_url": work_item.episode_url,
        "error": error_message,
    }


def _build_success_result_record(
    extraction_outcome: EpisodeExtractionOutcome,
) -> dict:
    return {
        "status": RESULT_STATUS_SUCCESS,
        "eid": extraction_outcome.eid,
        "title": extraction_outcome.title,
        "pub_date": extraction_outcome.pub_date,
        "episode_url": extraction_outcome.episode_url,
        "companies": [
            company.to_dict()
            for company in extraction_outcome.extraction_result.companies
        ],
        "filtered_count": extraction_outcome.extraction_result.filtered_count,
    }


def _get_optional_rate_env(env_name: str) -> float | None:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return None

    normalized_value = raw_value.strip()
    if not normalized_value:
        return None

    try:
        parsed_value = float(normalized_value)
    except ValueError as error:
        raise PipelineRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
        ) from error

    _validate_optional_rate(parsed_value, env_name)
    return parsed_value


def _validate_optional_rate(rate_per_minute: float | None, env_name: str) -> None:
    if rate_per_minute is None:
        return
    if not math.isfinite(rate_per_minute) or rate_per_minute <= 0:
        raise PipelineRateConfigError(
            INVALID_RATE_ENV_TEMPLATE.format(env_name=env_name)
        )


def _format_rate(rate_per_minute: float | None) -> str:
    if rate_per_minute is None:
        return "不限速"
    return f"{rate_per_minute}/分钟"
