from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Executor, Future, wait
from dataclasses import dataclass
from typing import Final, Sequence

from podcast_job_finder.companies.checkpoint import LlmCheckpointStore
from podcast_job_finder.companies.extraction import normalize_company_mentions
from podcast_job_finder.companies.models import (
    CompanyExtractionError,
    CompanyExtractionResult,
)
from podcast_job_finder.companies.runtime import EpisodeExtractionRuntime
from podcast_job_finder.companies.transcript_chunks import (
    TranscriptChunk,
    build_transcript_chunks,
)
from podcast_job_finder.companies.transcript_extraction import (
    INCOMPLETE_EXTRACTION_ERROR,
    TranscriptExtractionOutcome,
    _ExtractionExecutionContext,
    _extract_transcript_chunk,
    _merge_company_candidates,
)
from podcast_job_finder.episode.models import EpisodeWorkItem
from podcast_job_finder.llm import OpenAiCompatibleLlmError
from podcast_job_finder.transcription.models import TranscribedSpeechSegment


INCOMPLETE_CHUNK_RESULTS_ERROR: Final = "音频公司提取未生成完整的文本块结果。"
CANCELLED_CHUNK_ERROR: Final = "音频公司提取文本块任务被取消。"
EXPECTED_EXTRACTION_TASK_ERRORS: Final = (
    CompanyExtractionError,
    OpenAiCompatibleLlmError,
    OSError,
    ValueError,
)

type _LlmExtractionOutcome = tuple[CompanyExtractionResult, bool]
type TranscriptExtractionExecutionResult = TranscriptExtractionOutcome | Exception


@dataclass(slots=True, frozen=True)
class TranscriptExtractionRequest:
    work_item: EpisodeWorkItem
    title: str
    segments: tuple[TranscribedSpeechSegment, ...]
    checkpoint_store: LlmCheckpointStore


@dataclass(slots=True)
class _PendingTranscriptExtraction:
    """保存一个节目从提交文本块任务到生成节目结果期间的临时状态。

    context 保存该节目各次 LLM 请求共用的运行配置和检查点目录；chunks 按转写
    文本中的原始顺序保存所有文本块。每个文本块成功后，结果以 chunk.index 为键
    写入 chunk_outcomes；remaining_chunk_count 则记录还有多少文本块任务尚未由
    调度器处理，包括尚未提交、已经开始以及仍在线程池中等待的任务。

    first_error 只保存这个节目最先出现的错误。出现错误后，尚未提交的文本块会丢弃，
    已经提交但尚未开始的任务会尝试取消，已经开始的任务仍会等待并接收结果；数量
    归零后直接返回该错误，不会使用部分成功结果进行公司合并。没有错误时，调度器会
    先检查所有文本块结果是否完整，再按原始顺序合并候选公司。
    """

    context: _ExtractionExecutionContext
    title: str
    chunks: tuple[TranscriptChunk, ...]
    unscheduled_chunks: deque[TranscriptChunk]
    chunk_outcomes: dict[int, _LlmExtractionOutcome]
    remaining_chunk_count: int
    first_error: Exception | None = None


@dataclass(slots=True, frozen=True)
class _SubmittedChunkWork:
    """记录一个后台文本块任务属于哪个节目和哪个文本块。

    线程池返回的 Future 只用于判断任务是否完成和读取结果，本身不能说明结果属于
    哪个节目。episode_index 对应输入 requests 中的节目位置，调度器用它找到该节目
    的临时状态，并把最终结果写回同一位置。chunk_index 保存文本块下标，用于按原始
    顺序保存结果；节目失败时，这两个字段还用于只取消该节目的其他文本块任务。
    """

    episode_index: int
    chunk_index: int


def extract_companies_from_transcripts(
    *,
    requests: Sequence[TranscriptExtractionRequest],
    runtime: EpisodeExtractionRuntime,
    request_executor: Executor,
) -> list[TranscriptExtractionExecutionResult]:
    scheduler = _TranscriptExtractionScheduler(
        request_count=len(requests),
        runtime=runtime,
        request_executor=request_executor,
    )
    return scheduler.run(requests)


# 负责安排多个节目的公司提取任务，并把结果放回与输入请求相同的位置。
#
# 所有节目的文本块都会交给同一个线程池处理。某个节目的文本块全部成功时，调度器
# 按原来的文本块顺序收集候选公司；候选公司需要合并时，再提交一次合并任务。某个
# 节目失败只会记录在该节目的结果中，不会阻止其他节目继续处理。
class _TranscriptExtractionScheduler:
    def __init__(
        self,
        *,
        request_count: int,
        runtime: EpisodeExtractionRuntime,
        request_executor: Executor,
    ) -> None:
        self._runtime = runtime
        self._request_executor = request_executor
        # 先按请求数量占好位置。后台任务可以乱序完成，但写回时仍使用原请求下标，
        # 因此 run() 返回的结果顺序始终与 requests 相同。
        self._results: list[TranscriptExtractionExecutionResult | None] = [
            None
        ] * request_count
        # 保存每个已经启动的节目所需的文本块、已完成结果和第一个错误。
        self._states: dict[int, _PendingTranscriptExtraction] = {}
        # 文本块任务需要逐个接收结果，以便在一个节目首次失败时取消尚未开始的
        # 同节目任务。合并任务单独保存，不再与文本块任务共用阶段标记。
        self._pending_chunk_work: dict[
            Future[_LlmExtractionOutcome], _SubmittedChunkWork
        ] = {}
        self._merge_futures: dict[Future[_LlmExtractionOutcome], int] = {}
        # 每个有待提交文本块的节目在队列中最多出现一次。每次只取一个文本块，
        # 仍有剩余时把节目放回队尾，让长节目不会挡住后续节目。
        self._ready_episode_indexes: deque[int] = deque()

    def run(
        self,
        requests: Sequence[TranscriptExtractionRequest],
    ) -> list[TranscriptExtractionExecutionResult]:
        # 调用方已经保证每个节目使用独立检查点目录，因此可以登记所有节目，
        # 再由轮转队列按任务上限逐步提交文本块。
        for episode_index, request in enumerate(requests):
            self._submit_request(episode_index, request)
        self._collect_results()
        # 正常情况下，每个请求都会得到成功结果或错误。这里用于发现调度流程
        # 提前结束、导致某个位置没有被写入的程序错误。
        if any(result is None for result in self._results):
            raise RuntimeError(INCOMPLETE_EXTRACTION_ERROR)
        return [result for result in self._results if result is not None]

    def _submit_request(
        self,
        episode_index: int,
        request: TranscriptExtractionRequest,
    ) -> None:
        chunks = tuple(build_transcript_chunks(request.segments))
        if not chunks:
            self._results[episode_index] = TranscriptExtractionOutcome(
                extraction_result=CompanyExtractionResult(),
                chunk_count=0,
                candidate_count=0,
                cached=False,
            )
            return

        # 一个节目内的所有文本块共享执行配置，但分别保存完成结果。后台任务的
        # 完成顺序不固定，所以结果以 chunk.index 为键保存，汇总时再恢复原顺序。
        state = _PendingTranscriptExtraction(
            context=_ExtractionExecutionContext(
                work_item=request.work_item,
                runtime=self._runtime,
                checkpoint_store=request.checkpoint_store,
            ),
            title=request.title,
            chunks=chunks,
            unscheduled_chunks=deque(chunks),
            chunk_outcomes={},
            remaining_chunk_count=len(chunks),
        )
        self._states[episode_index] = state
        self._ready_episode_indexes.append(episode_index)

    def _fill_available_slots(self) -> None:
        while (
            self._ready_episode_indexes
            and self._pending_task_count() < self._runtime.llm.max_in_flight_requests
        ):
            episode_index = self._ready_episode_indexes.popleft()
            state = self._states[episode_index]
            chunk = state.unscheduled_chunks.popleft()
            future = self._request_executor.submit(
                _extract_transcript_chunk,
                chunk=chunk,
                title=state.title,
                context=state.context,
            )
            self._pending_chunk_work[future] = _SubmittedChunkWork(
                episode_index=episode_index,
                chunk_index=chunk.index,
            )
            if state.unscheduled_chunks:
                self._ready_episode_indexes.append(episode_index)

    def _pending_task_count(self) -> int:
        return len(self._pending_chunk_work) + len(self._merge_futures)

    def _collect_results(self) -> None:
        while (
            self._pending_chunk_work
            or self._merge_futures
            or self._ready_episode_indexes
        ):
            self._fill_available_slots()
            pending_futures = set(self._pending_chunk_work) | set(self._merge_futures)
            if not pending_futures:
                break
            # 任意文本块或合并任务完成就立即处理，并按空出的名额轮转补充文本块。
            completed_futures, _ = wait(
                pending_futures,
                return_when=FIRST_COMPLETED,
            )
            for future in completed_futures:
                if future in self._pending_chunk_work:
                    self._process_completed_chunk(future)
                else:
                    self._process_completed_merge(future)

    def _process_completed_chunk(
        self,
        future: Future[_LlmExtractionOutcome],
    ) -> None:
        work = self._pending_chunk_work.pop(future)
        state = self._states[work.episode_index]
        had_error = state.first_error is not None
        self._record_chunk_result(future, work.chunk_index, state)
        # 只在这个任务首次写入节目错误时取消一次。之后取回已取消的任务时，
        # first_error 已经存在，无需再次遍历尚未完成的任务。
        if not had_error and state.first_error is not None:
            self._cancel_remaining_chunks(work.episode_index, state)
        if state.remaining_chunk_count == 0:
            self._finish_chunk_stage(work.episode_index, state)

    def _record_chunk_result(
        self,
        future: Future[_LlmExtractionOutcome],
        chunk_index: int,
        state: _PendingTranscriptExtraction,
    ) -> None:
        # 被取消的任务也算已经结束，否则节目会一直等待一个不会再返回的任务。
        state.remaining_chunk_count -= 1
        if future.cancelled():
            if state.first_error is None:
                state.first_error = RuntimeError(CANCELLED_CHUNK_ERROR)
            return
        # future.result() 会在后台任务失败时重新抛出异常。这里只保存当前节目的
        # 第一个错误，稍后写入这个节目的结果，其他节目仍然可以继续处理。
        try:
            state.chunk_outcomes[chunk_index] = future.result()
        except EXPECTED_EXTRACTION_TASK_ERRORS as error:
            if state.first_error is None:
                state.first_error = error

    def _cancel_remaining_chunks(
        self,
        episode_index: int,
        state: _PendingTranscriptExtraction,
    ) -> None:
        state.remaining_chunk_count -= len(state.unscheduled_chunks)
        state.unscheduled_chunks.clear()
        try:
            self._ready_episode_indexes.remove(episode_index)
        except ValueError:
            pass
        # cancel() 只能取消还没在线程中开始的任务；已经开始的任务仍会执行完，
        # 并由等待循环正常取回，这样线程池里不会留下无人处理的结果。
        for future, work in self._pending_chunk_work.items():
            if work.episode_index == episode_index:
                future.cancel()

    def _finish_chunk_stage(
        self,
        episode_index: int,
        state: _PendingTranscriptExtraction,
    ) -> None:
        # 只有当前节目的全部文本块都已结束，才会进入这里。失败时直接记录错误，
        # 不再使用可能已经成功返回的部分结果。
        if state.first_error is not None:
            self._results[episode_index] = state.first_error
            return

        expected_chunk_indexes = {chunk.index for chunk in state.chunks}
        # remaining_chunk_count 归零只能证明所有文本块任务都已经结束并被调度器
        # 接收，不能单独证明每个成功结果都已写入 chunk_outcomes。例如节目包含下标
        # {1, 2, 3} 的三个文本块时，chunk_outcomes 也必须正好包含这三个下标；缺少
        # 某个下标或出现其他下标都表示内部记录不完整，不能继续合并候选公司。
        if set(state.chunk_outcomes) != expected_chunk_indexes:
            self._results[episode_index] = RuntimeError(INCOMPLETE_CHUNK_RESULTS_ERROR)
            return

        # 后台任务可能乱序完成，这里按原文本块顺序收集候选公司，保证后续合并
        # 接收到稳定的输入顺序。
        ordered_outcomes = self._ordered_chunk_outcomes(state)
        candidates = [
            company
            for chunk_result, _ in ordered_outcomes
            for company in chunk_result.companies
        ]
        # 只有一个文本块时没有跨块重复需要判断；没有候选公司时也无需再请求
        # LLM。两种情况都只在本地应用公司黑名单和去重规则。
        if len(state.chunks) == 1 or not candidates:
            extraction_result = normalize_company_mentions(
                candidates,
                company_blacklist=state.context.runtime.company_blacklist,
            )
            self._results[episode_index] = TranscriptExtractionOutcome(
                extraction_result=extraction_result,
                chunk_count=len(state.chunks),
                candidate_count=len(candidates),
                cached=all(cached for _, cached in ordered_outcomes),
            )
            return

        # 节目被拆成多个文本块，并且这些文本块合计至少找到一个候选公司时，
        # 再让 LLM 判断跨文本块出现的名称是否指向同一家公司。
        merge_future = self._request_executor.submit(
            _merge_company_candidates,
            candidates=candidates,
            context=state.context,
        )
        self._merge_futures[merge_future] = episode_index

    def _process_completed_merge(
        self,
        future: Future[_LlmExtractionOutcome],
    ) -> None:
        episode_index = self._merge_futures.pop(future)
        self._record_merge_result(
            future,
            episode_index,
            self._states[episode_index],
        )

    def _record_merge_result(
        self,
        future: Future[_LlmExtractionOutcome],
        episode_index: int,
        state: _PendingTranscriptExtraction,
    ) -> None:
        # future.result() 会在合并失败时重新抛出异常。该异常只写入当前节目的
        # 结果，其他节目仍会继续处理。
        try:
            extraction_result, merge_cached = future.result()
        except EXPECTED_EXTRACTION_TASK_ERRORS as error:
            self._results[episode_index] = error
            return

        ordered_outcomes = self._ordered_chunk_outcomes(state)
        self._results[episode_index] = TranscriptExtractionOutcome(
            extraction_result=extraction_result,
            chunk_count=len(state.chunks),
            candidate_count=sum(
                len(chunk_result.companies) for chunk_result, _ in ordered_outcomes
            ),
            cached=(all(cached for _, cached in ordered_outcomes) and merge_cached),
        )

    @staticmethod
    def _ordered_chunk_outcomes(
        state: _PendingTranscriptExtraction,
    ) -> list[_LlmExtractionOutcome]:
        """按文本块在转写中的原始顺序取回后台任务结果。

        文本块请求可能以不同于提交顺序的先后次序完成，因此 chunk_outcomes
        的写入顺序不能代表原始文本顺序。这里依次读取 state.chunks 中保存的
        chunk.index，创建顺序稳定的新列表，供候选公司合并和缓存状态统计使用；
        原来的 chunk_outcomes 字典不会被修改。
        """
        return [state.chunk_outcomes[chunk.index] for chunk in state.chunks]
