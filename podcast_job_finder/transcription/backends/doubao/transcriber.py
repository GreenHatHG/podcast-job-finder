"""豆包音频转写的主流程。

这个文件负责把一个音频片段从“发送给豆包”处理到“整理成项目可以保存的结果”。
它会重试空结果，并在识别结果疑似漏内容时检查可疑位置、决定是否完整重试。
"""

# pylint: disable=too-many-lines

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
import wave
from collections import deque
from concurrent.futures import CancelledError, Future
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Iterator, Sequence

from podcast_job_finder.audio.segmentation.segment_export import (
    WAV_SEGMENT_AUDIO_FORMAT,
    ExportedSpeechSegment,
)
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    detect_and_export_speech_segments,
)
from podcast_job_finder.audio.segmentation.vad import (
    VAD_SAMPLE_RATE,
    SpeechSegment,
    VadConfig,
)
from podcast_job_finder.transcription.backends.firered.alignment import (
    FireRedTextAlignmentClient,
)
from podcast_job_finder.transcription.models import (
    AudioTranscriptionError,
    CharacterAlignment,
    TranscriptionSegmentResult,
    TranscribedSpeechSegment,
    TranscriptionOutput,
)
from podcast_job_finder.transcription.segments import (
    build_transcribed_speech_segment,
    validate_previous_segment_order,
)
from podcast_job_finder.transcription.diagnostics import (
    TranscriptionAttemptDiagnostics,
    TranscriptionDiagnostics,
    TruncationProbeDiagnostics,
    TruncationAssessment,
)
from podcast_job_finder.transcription.quality import (
    assess_transcription_coverage,
)
from podcast_job_finder.timestamps import format_duration_ms
from podcast_job_finder.audio.segmentation.speech_pipeline import (
    DEFAULT_SILENCE_PADDING_MS,
)
from podcast_job_finder.audio.segmentation.vad import DEFAULT_VAD_THRESHOLD

from .client import build_doubao_client
from .config import (
    DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS,
    DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS,
    DOUBAO_MISSING_FINAL_SEGMENT_THRESHOLD,
    DOUBAO_RESPONSE_MAX_ATTEMPTS,
    DOUBAO_RETRY_BASE_DELAY_SECONDS,
)
from .output import (
    DoubaoOverlapOnlyResultError,
    build_doubao_transcription_output,
)
from .request_client import (
    DoubaoJob,
    DoubaoRequestError,
    SessionResponses,
    run_doubao_job,
)
from .request_scheduler import DoubaoRequestScheduler
from .response import (
    AsrResponseProtocol,
    DOUBAO_RESPONSE_ERROR,
    DoubaoMissingFinalResponseError,
    DoubaoResponseSummary,
    build_doubao_response_summary,
)
from .truncation_probe import (
    TruncationProbeRequest,
    log_truncation_probe_result,
    probe_requires_full_retry,
    run_doubao_truncation_probe,
)


logger = logging.getLogger(__name__)
DOUBAO_FALLBACK_TRAILING_SILENCE_MS = 2_000
DOUBAO_OVERLAP_FALLBACK_SPLIT_COUNT = 2
DOUBAO_ASR_SERVICE_ERROR = (
    "豆包 ASR 在 {count} 个不同音频片段发生服务或协议异常，达到停止阈值："
    "threshold={threshold} latest_path={path} latest_reason={reason}"
)


def _describe_doubao_job(job: DoubaoJob) -> str:
    if job.segment is None or job.total_segment_count is None:
        return f"path={job.path} type=probe_or_retry"
    segment = job.segment.segment
    return (
        f"index={job.segment.index}/{job.total_segment_count} "
        f"path={job.path} type=segment "
        f"start={format_duration_ms(segment.start_ms)} "
        f"end={format_duration_ms(segment.end_ms)} "
        f"duration={format_duration_ms(segment.duration_ms)}"
    )


class DoubaoServiceErrorThresholdExceeded(AudioTranscriptionError):
    """豆包服务异常片段达到阈值，本批次不能继续发送。"""


@dataclass(slots=True, frozen=True)
class _AttemptResult:
    """记录一次识别尝试的完整结果。

    同一个片段可能因为异常空响应尝试多次，也可能在发现“识别结果里有一段人声
    没有对应文字”后进行一次完整重试。这里把文字、每个字出现的时间和检查结果
    放在一起，方便最后选择更可靠的一次。
    """

    # 第几次识别，从 1 开始。
    attempt: int
    # 从豆包返回的多条消息中整理出的文字和检查结果。
    response_summary: DoubaoResponseSummary
    # 每个字在音频中的开始、结束时间，以及程序认为它识别得有多可靠。
    alignments: tuple[CharacterAlignment, ...]
    # 用来判断是否有一段人声没有对应文字。
    assessment: TruncationAssessment
    # 这次请求无法得到正常结果时，记录具体原因供检查点诊断使用。
    error: str | None = None

    @property
    def mean_confidence(self) -> float:
        """计算这次识别中所有文字的平均置信度。"""

        # 没有文字时没有可计算的置信度，按 0 处理。
        if not self.alignments:
            return 0.0
        return sum(item.confidence for item in self.alignments) / len(self.alignments)


@dataclass(slots=True)
class _SegmentAttempts:
    """保存一个片段的识别尝试和可选的额外检查结果。"""

    # 保存可用于输出和诊断的尝试；中间缺少最终响应的请求不会单独保存。
    attempts: list[_AttemptResult]
    # 只有第一次结果可疑时才会有这个值。
    probe: TruncationProbeDiagnostics | None = None


@dataclass(slots=True, frozen=True)
class _PendingRequest:
    """保存一个已经提交、正在等待结果的请求。"""

    # 请求对应的音频片段。
    segment: ExportedSpeechSegment
    # 后台线程完成后，从 Future 中取出豆包响应。
    future: Future[SessionResponses]


class DoubaoAudioTranscriber:  # pylint: disable=too-many-instance-attributes
    """负责请求豆包、检查结果，并整理出带时间信息的文字。"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        aligner: FireRedTextAlignmentClient,
        max_in_flight_requests: int = DEFAULT_DOUBAO_MAX_IN_FLIGHT_REQUESTS,
        request_interval_seconds: float = DEFAULT_DOUBAO_REQUEST_INTERVAL_SECONDS,
        silence_padding_ms: int = DEFAULT_SILENCE_PADDING_MS,
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
    ) -> None:
        """创建转写器及其所需的辅助组件。"""

        if max_in_flight_requests <= 0:
            raise ValueError("DOUBAO_ASR_MAX_IN_FLIGHT_REQUESTS 必须大于 0。")
        if not math.isfinite(request_interval_seconds) or request_interval_seconds <= 0:
            raise ValueError("request_interval_seconds 必须是大于 0 的有限数值。")
        if silence_padding_ms < 0:
            raise ValueError("silence_padding_ms 必须大于等于 0。")
        if not 0 < vad_threshold < 1:
            raise ValueError("vad_threshold 必须大于 0 且小于 1。")

        self._max_in_flight_requests = max_in_flight_requests
        self._silence_padding_ms = silence_padding_ms
        self._vad_threshold = vad_threshold
        self._service_error_segment_paths: set[Path] = set()
        self._service_error_lock = Lock()
        self._service_error_threshold_error: (
            DoubaoServiceErrorThresholdExceeded | None
        ) = None
        self._aligner = aligner

        # 拿到豆包识别函数和响应类型。识别函数交给调度器，响应类型用来判断最终结果。
        (
            asr_config,
            transcribe_stream,
            self._response_type,
        ) = build_doubao_client()
        self._request_scheduler = DoubaoRequestScheduler(
            max_in_flight_requests=max_in_flight_requests,
            interval_seconds=request_interval_seconds,
            worker=partial(
                run_doubao_job,
                transcribe_stream=transcribe_stream,
                asr_config=asr_config,
            ),
            request_description=_describe_doubao_job,
        )
        logger.info(
            "豆包请求调度配置：request_interval_seconds=%.3f max_in_flight_requests=%d",
            request_interval_seconds,
            max_in_flight_requests,
        )

    @property
    def batch_size(self) -> int:
        """告诉上层一次最多可以同时等待多少个豆包请求。"""

        return self._max_in_flight_requests

    def _submit_segment(
        self,
        segment: ExportedSpeechSegment,
        *,
        total_segment_count: int,
    ) -> Future[SessionResponses]:
        with self._service_error_lock:
            threshold_error = getattr(
                self,
                "_service_error_threshold_error",
                None,
            )
            if threshold_error is not None:
                raise threshold_error
            future = self._request_scheduler.submit(
                DoubaoJob(
                    path=segment.file_path,
                    segment=segment,
                    total_segment_count=total_segment_count,
                )
            )
        future.add_done_callback(partial(self._record_completed_request_error, segment))
        return future

    def _record_completed_request_error(
        self,
        segment: ExportedSpeechSegment,
        future: Future[SessionResponses],
    ) -> None:
        """请求完成时立即统计豆包服务异常，不等待上层按片段顺序读取。"""

        try:
            error = future.exception()
        except CancelledError:
            return
        if not isinstance(error, DoubaoRequestError):
            return
        try:
            self._record_service_error_segment(segment, reason=str(error))
        except DoubaoServiceErrorThresholdExceeded:
            self._request_scheduler.cancel_pending()

    def _recognize_path(self, path: Path) -> SessionResponses:
        return self._request_scheduler.submit(
            DoubaoJob(path=path),
            high_priority=True,
        ).result()

    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        """按顺序处理一个片段，并返回可保存的转写结果。

        处理顺序是：初次识别、必要时重试空结果、检查是否有一段人声没有对应
        文字、必要时完整重试，最后去除和前一个片段重复的内容并生成时间信息。
        """

        # 先完成一次正常识别，后面的判断都基于这次结果进行。
        try:
            initial_responses = self._recognize_path(segment.file_path)
            attempts = self._run_segment_attempts_with_fallback(
                segment,
                initial_responses,
            )
        except DoubaoRequestError as error:
            self._record_service_error_segment(segment, reason=str(error))
            raise

        # 选择更完整可靠的结果，并换算成它在原始音频中的实际时间。
        try:
            return self._build_transcription_output_with_fallback(
                segment,
                attempts,
                previous_segment=previous_segment,
            )
        except DoubaoRequestError as error:
            self._record_service_error_segment(segment, reason=str(error))
            raise

    def transcribe_batches(
        self,
        segments: Sequence[ExportedSpeechSegment],
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> Iterator[Sequence[TranscriptionSegmentResult]]:
        """并发提交多个片段，同时按原始顺序逐个产出结果。

        这里表示同时推进多个请求，单个片段的识别和重试仍由
        `_run_segment_attempts` 负责。按顺序取结果是为了处理相邻片段中重复的
        那小段音频：当前片段需要知道前一个片段已经保存到哪里。
        """

        # 没有片段时直接结束，不产生任何结果。
        if not segments:
            return

        # 按提交顺序保存正在等待结果的请求。
        pending_requests: deque[_PendingRequest] = deque()
        current_previous = previous_segment
        total_segment_count = max(segment.index for segment in segments)
        # 先把这一组片段全部交给调度器排队。发送间隔由调度器控制，
        # 不需要等当前片段识别或对齐结束才提交下一段。
        for segment in segments:
            future = self._submit_segment(
                segment,
                total_segment_count=total_segment_count,
            )
            pending_requests.append(_PendingRequest(segment, future))

        first_error: Exception | None = None
        skip_next_successful_result = False
        try:
            while pending_requests:
                pending_request = pending_requests.popleft()
                item, current_previous, error = self._consume_pending_request(
                    pending_request,
                    previous_segment=current_previous,
                )
                if error is not None:
                    if isinstance(error, DoubaoServiceErrorThresholdExceeded):
                        raise error
                    if first_error is None:
                        first_error = error
                    # 下一片段依赖当前失败片段的识别结果去除重叠内容。即使下一片段
                    # 请求成功，它的结果也不可靠，因此只用来恢复再下一片段的依赖。
                    skip_next_successful_result = True
                    continue
                if item is not None and skip_next_successful_result:
                    logger.warning(
                        "上一音频片段失败，当前片段只用于恢复后续依赖，不写入检查点："
                        "index=%d path=%s",
                        item.segment.index,
                        item.segment.file_path,
                    )
                    skip_next_successful_result = False
                elif item is not None:
                    # 成功结果马上交给上层，上层可以立刻写入片段检查点。
                    yield (item,)
        finally:
            # 上层保存检查点失败或提前关闭生成器时，停止尚未发送的请求。
            self._request_scheduler.cancel_pending()

        # 后续片段全部处理完后，再把最早的错误报告给上层。
        if first_error is not None:
            raise first_error

    def _consume_pending_request(
        self,
        pending_request: _PendingRequest,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> tuple[
        TranscriptionSegmentResult | None,
        TranscribedSpeechSegment | None,
        Exception | None,
    ]:
        """读取一个请求的结果，并把异常转换成可继续处理的状态。

        返回值依次表示：转写结果、供下一个片段使用的前一片段信息、错误。
        这样即使某个请求失败，也能先处理其它已经在途并成功返回的请求。
        """
        segment = pending_request.segment
        logger.info(
            "豆包片段结果读取开始：index=%d path=%s request_ready=%s",
            segment.index,
            segment.file_path,
            pending_request.future.done(),
        )
        request_wait_started_at = time.monotonic()
        try:
            # 等待后台请求完成，然后取得豆包这次返回的全部消息。
            responses = pending_request.future.result()
        except AudioTranscriptionError as error:
            # 后续片段只需要失败片段的编号和结束时间来去除重叠音频。
            return (
                None,
                _build_failed_segment_context(segment),
                self._reported_request_error(segment, error),
            )
        logger.info(
            "豆包片段结果读取完成：index=%d path=%s elapsed_seconds=%.3f",
            segment.index,
            segment.file_path,
            time.monotonic() - request_wait_started_at,
        )
        processing_started_at = time.monotonic()
        logger.info(
            "豆包片段结果整理开始：index=%d path=%s",
            segment.index,
            segment.file_path,
        )
        try:
            # 整理返回消息，检查是否有一段人声没有对应文字，并在需要时重试。
            validate_previous_segment_order(
                segment,
                previous_segment,
            )
            item = self._build_transcription_item(
                segment,
                responses,
                previous_segment=previous_segment,
            )
        except AudioTranscriptionError as error:
            logger.info(
                "豆包片段结果整理失败：index=%d path=%s elapsed_seconds=%.3f "
                "error_type=%s",
                segment.index,
                segment.file_path,
                time.monotonic() - processing_started_at,
                type(error).__name__,
            )
            return (
                None,
                _build_failed_segment_context(segment),
                self._reported_request_error(segment, error),
            )
        logger.info(
            "豆包片段结果整理完成：index=%d path=%s elapsed_seconds=%.3f",
            segment.index,
            segment.file_path,
            time.monotonic() - processing_started_at,
        )
        return (
            item,
            # 下一片段需要使用当前片段的完整结果作为参考。
            build_transcribed_speech_segment(segment, item.output),
            None,
        )

    def _build_transcription_item(
        self,
        segment: ExportedSpeechSegment,
        responses: SessionResponses,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionSegmentResult:
        """把一个片段的豆包返回消息整理成上层可以保存的片段结果。"""

        attempts = self._run_segment_attempts_with_fallback(segment, responses)
        return TranscriptionSegmentResult(
            segment=segment,
            output=self._build_transcription_output_with_fallback(
                segment,
                attempts,
                previous_segment=previous_segment,
            ),
            previous_segment=previous_segment,
        )

    def _build_transcription_output_with_fallback(
        self,
        segment: ExportedSpeechSegment,
        attempts: _SegmentAttempts,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        """整段结果只有重叠文字时，重切当前片段并分别识别。"""

        try:
            return self._build_transcription_output(
                segment,
                attempts,
                previous_segment=previous_segment,
            )
        except DoubaoOverlapOnlyResultError as error:
            logger.warning(
                "豆包只识别到重叠音频，将临时重切片段后重试：index=%d path=%s",
                segment.index,
                segment.file_path,
            )
            return self._transcribe_resplit_segment(
                segment,
                previous_segment=previous_segment,
                original_error=error,
            )

    def _transcribe_resplit_segment(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
        original_error: DoubaoOverlapOnlyResultError,
    ) -> TranscriptionOutput:
        """使用项目 VAD 重切失败片段，并把各子片段结果合并回一个输出。"""

        with _temporary_resplit_segments(
            segment,
            silence_padding_ms=self._silence_padding_ms,
            vad_threshold=self._vad_threshold,
        ) as child_segments:
            if len(child_segments) < DOUBAO_OVERLAP_FALLBACK_SPLIT_COUNT:
                raise original_error

            outputs: list[TranscriptionOutput] = []
            current_previous = previous_segment
            for child_segment in child_segments:
                responses = self._recognize_path(child_segment.file_path)
                child_attempts = self._run_segment_attempts_with_fallback(
                    child_segment,
                    responses,
                )
                try:
                    output = self._build_transcription_output(
                        child_segment,
                        child_attempts,
                        previous_segment=current_previous,
                    )
                except DoubaoOverlapOnlyResultError:
                    continue
                if not output.text:
                    continue
                outputs.append(output)
                current_previous = build_transcribed_speech_segment(
                    child_segment,
                    output,
                )

        if not outputs:
            raise original_error
        logger.info(
            "豆包重切片段识别完成：index=%d child_segments=%d retained_outputs=%d "
            "path=%s",
            segment.index,
            len(child_segments),
            len(outputs),
            segment.file_path,
        )
        return _merge_transcription_outputs(outputs)

    def _run_segment_attempts_with_fallback(
        self,
        segment: ExportedSpeechSegment,
        initial_responses: SessionResponses,
    ) -> _SegmentAttempts:
        """在缺少终止响应时追加尾部静音，并重新请求当前片段。"""

        try:
            return self._run_segment_attempts(segment, initial_responses)
        except DoubaoMissingFinalResponseError as error:
            logger.warning(
                "豆包片段缺少完整终止响应，将追加 %dms 尾部静音后重试：index=%d path=%s",
                DOUBAO_FALLBACK_TRAILING_SILENCE_MS,
                segment.index,
                segment.file_path,
            )
            try:
                with _temporary_trailing_silence(
                    segment.file_path,
                    duration_ms=DOUBAO_FALLBACK_TRAILING_SILENCE_MS,
                ) as padded_path:
                    padded_segment = ExportedSpeechSegment(
                        index=segment.index,
                        segment=segment.segment,
                        file_path=padded_path,
                    )
                    padded_responses = self._recognize_path(padded_path)
                    return self._run_segment_attempts(
                        padded_segment,
                        padded_responses,
                    )
            except (OSError, wave.Error) as fallback_error:
                raise error from fallback_error

    def _run_segment_attempts(
        self,
        segment: ExportedSpeechSegment,
        initial_responses: SessionResponses,
    ) -> _SegmentAttempts:
        """完成一个片段从初次识别到可选重试的完整流程。

        初次请求没有最终消息或文字为空时按计划等待并重试。得到文字后，如果
        发现有一段人声没有对应文字，就单独识别那段音频；确认那里确实有人说话，
        或单独识别过程发生错误时，再完整重试当前片段。
        """

        attempts = self._run_response_attempts(segment, initial_responses)
        latest_attempt = attempts.attempts[-1]

        # 已经用完异常空响应重试次数时，保留豆包实际返回的空结果。
        if not latest_attempt.response_summary.text:
            return attempts

        # 结果正常时不做额外请求，避免增加服务压力和处理时间。
        if not latest_attempt.assessment.requires_truncation_probe:
            return attempts

        # 单独识别没有对应文字的那段音频，确认那里是否真的有人说话。
        attempts.probe = run_doubao_truncation_probe(
            TruncationProbeRequest(
                audio_path=segment.file_path,
                assessment=latest_attempt.assessment,
            ),
            collect_responses=self._recognize_path,
            response_types=self._response_type,
        )
        log_truncation_probe_result(
            segment_index=segment.index,
            assessment=latest_attempt.assessment,
            probe=attempts.probe,
        )
        if probe_requires_full_retry(attempts.probe):
            # 探测确认需要重试时，重新识别完整片段，而不是只识别局部音频。
            attempts.attempts.append(
                self._run_attempt(segment, attempt=latest_attempt.attempt + 1)
            )
        return attempts

    def _run_response_attempts(
        self,
        segment: ExportedSpeechSegment,
        initial_responses: SessionResponses,
    ) -> _SegmentAttempts:
        """重试缺少最终响应或文字为空的请求，直到得到非空文字。"""

        completed_attempts: list[_AttemptResult] = []
        last_missing_final_error: DoubaoMissingFinalResponseError | None = None
        for attempt in range(1, DOUBAO_RESPONSE_MAX_ATTEMPTS + 1):
            try:
                attempt_result = (
                    self._build_attempt_result(
                        segment,
                        initial_responses,
                        attempt=attempt,
                    )
                    if attempt == 1
                    else self._run_attempt(segment, attempt=attempt)
                )
            except DoubaoMissingFinalResponseError as error:
                last_missing_final_error = error
                retry_reason = "未返回完整终止响应"
            else:
                completed_attempts.append(attempt_result)
                if attempt_result.response_summary.text:
                    return _SegmentAttempts(completed_attempts)
                retry_reason = "返回空文字"

            if attempt < DOUBAO_RESPONSE_MAX_ATTEMPTS:
                self._wait_before_response_retry(
                    segment,
                    attempt=attempt,
                    reason=retry_reason,
                )

        if completed_attempts:
            return _SegmentAttempts(completed_attempts)
        # 每次请求要么加入 completed_attempts，要么记录缺少最终响应的错误。
        # 运行到这里却两者都没有，说明重试流程本身出现了未预期的状态。
        # 防止以后修改重试逻辑时产生“不执行任何请求，也没有记录错误”的异常状态
        if last_missing_final_error is None:
            raise AssertionError("豆包异常响应重试结束后没有可用结果或错误。")
        self._record_service_error_segment(
            segment,
            reason=(f"连续 {DOUBAO_RESPONSE_MAX_ATTEMPTS} 次未返回完整终止响应"),
        )
        raise last_missing_final_error

    @staticmethod
    def _wait_before_response_retry(
        segment: ExportedSpeechSegment,
        *,
        attempt: int,
        reason: str,
    ) -> None:
        """按指数增长的等待时间暂停，然后重试异常响应。"""

        delay_seconds = DOUBAO_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
        logger.warning(
            "豆包片段%s，将在 %.1f 秒后重试：index=%d attempt=%d/%d "
            "next_attempt=%d/%d path=%s",
            reason,
            delay_seconds,
            segment.index,
            attempt,
            DOUBAO_RESPONSE_MAX_ATTEMPTS,
            attempt + 1,
            DOUBAO_RESPONSE_MAX_ATTEMPTS,
            segment.file_path,
        )
        time.sleep(delay_seconds)

    def _build_transcription_output(
        self,
        segment: ExportedSpeechSegment,
        attempts: _SegmentAttempts,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        """从所有尝试中选择较可靠的一次，并生成最终输出。"""

        # 优先选择更多人声有对应文字、遗漏时长更短、识别把握更高的结果。
        selected = max(attempts.attempts, key=_attempt_quality_key)

        # 保存每次识别和额外检查的记录，方便之后查看问题片段。
        diagnostics = TranscriptionDiagnostics(
            selected_attempt=selected.attempt,
            attempts=tuple(
                _build_attempt_diagnostics(item) for item in attempts.attempts
            ),
            truncation_probe=attempts.probe,
        )

        # 去掉片段之间重复的内容，并生成每个字和每句话的时间信息。
        return build_doubao_transcription_output(
            selected.response_summary.text,
            selected.alignments,
            segment=segment,
            previous_segment=previous_segment,
            silence_padding_ms=self._silence_padding_ms,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        """关闭请求调度器和负责查找文字时间的进程。"""

        self._request_scheduler.close()
        self._aligner.close()

    def _run_attempt(
        self,
        segment: ExportedSpeechSegment,
        *,
        attempt: int,
    ) -> _AttemptResult:
        """重新请求整个片段，并整理这次请求的结果。"""

        responses = self._recognize_path(segment.file_path)
        return self._build_attempt_result(segment, responses, attempt=attempt)

    def _reported_request_error(
        self,
        segment: ExportedSpeechSegment,
        error: Exception,
    ) -> Exception:
        """把豆包请求异常计入阈值，并返回上层最终应报告的错误。"""

        if not isinstance(error, DoubaoRequestError):
            return error
        try:
            self._record_service_error_segment(segment, reason=str(error))
        except AudioTranscriptionError as service_error:
            return service_error
        return error

    def _record_service_error_segment(
        self,
        segment: ExportedSpeechSegment,
        *,
        reason: str,
    ) -> None:
        """累计豆包服务或协议异常片段，达到阈值时停止批处理。"""

        threshold_error: DoubaoServiceErrorThresholdExceeded | None = None
        with self._service_error_lock:
            self._service_error_segment_paths.add(segment.file_path)
            segment_count = len(self._service_error_segment_paths)
            threshold = DOUBAO_MISSING_FINAL_SEGMENT_THRESHOLD
            if segment_count >= threshold:
                threshold_error = DoubaoServiceErrorThresholdExceeded(
                    DOUBAO_ASR_SERVICE_ERROR.format(
                        count=segment_count,
                        threshold=threshold,
                        path=segment.file_path,
                        reason=reason,
                    )
                )
                self._service_error_threshold_error = threshold_error
        logger.warning(
            "豆包服务或协议异常片段：index=%d service_error_segments=%d "
            "threshold=%d path=%s reason=%s",
            segment.index,
            segment_count,
            threshold,
            segment.file_path,
            reason,
        )
        if segment_count < threshold:
            return
        assert threshold_error is not None
        raise threshold_error

    def _build_attempt_result(
        self,
        segment: ExportedSpeechSegment,
        responses: list[AsrResponseProtocol],
        *,
        attempt: int,
    ) -> _AttemptResult:
        """把豆包响应转换成一次完整的识别记录。

        这里会提取最终文字、确认豆包已经完成识别、找出每个字在音频中的时间，
        并检查是否有一段人声没有对应文字。后面的流程只使用这个整理结果。
        """

        # 一个请求会收到多条豆包消息，这里提取最终文字并保留检查所需的信息。
        response_summary = build_doubao_response_summary(
            responses,
            final_response_type=self._response_type.FINAL_RESULT,
            error_response_type=self._response_type.ERROR,
            terminal_response_type=self._response_type.SESSION_FINISHED,
        )
        # 豆包没有返回“识别完成”的消息时，这次结果无法使用。
        _validate_response_summary(response_summary, segment.file_path)
        if not response_summary.text:
            logger.info(
                "豆包片段未识别到文字，保存空结果：index=%d path=%s",
                segment.index,
                segment.file_path,
            )
        if response_summary.text:
            alignment_started_at = time.monotonic()
            logger.info(
                "FireRed CTC 对齐开始：index=%d path=%s text_length=%d",
                segment.index,
                segment.file_path,
                len(response_summary.text),
            )
            try:
                alignments = self._aligner.align(
                    segment.file_path,
                    response_summary.text,
                )
            except AudioTranscriptionError:
                logger.exception(
                    "FireRed CTC 对齐失败：index=%d path=%s elapsed_seconds=%.3f",
                    segment.index,
                    segment.file_path,
                    time.monotonic() - alignment_started_at,
                )
                raise
            logger.info(
                "FireRed CTC 对齐完成：index=%d path=%s elapsed_seconds=%.3f "
                "alignment_count=%d",
                segment.index,
                segment.file_path,
                time.monotonic() - alignment_started_at,
                len(alignments),
            )
        else:
            # 没有文字时无需查找每个字的时间，直接检查音频里是否有人说话。
            alignments = ()
        # 比较“有人说话的时间”和“文字出现的时间”，找出是否有一段人声没有文字。
        assessment = assess_transcription_coverage(
            segment.file_path,
            alignments,
            has_transcript=bool(response_summary.text),
            speech_segment=segment.segment,
            silence_padding_ms=self._silence_padding_ms,
            vad_threshold=self._vad_threshold,
        )
        return _AttemptResult(
            attempt=attempt,
            response_summary=response_summary,
            alignments=alignments,
            assessment=assessment,
        )


def _build_failed_segment_context(
    segment: ExportedSpeechSegment,
) -> TranscribedSpeechSegment:
    """为失败片段创建只用于处理后续片段的上下文。"""

    return TranscribedSpeechSegment(
        index=segment.index,
        start_ms=segment.segment.start_ms,
        end_ms=segment.segment.end_ms,
        text="",
    )


@contextmanager
def _temporary_resplit_segments(
    source_segment: ExportedSpeechSegment,
    *,
    silence_padding_ms: int,
    vad_threshold: float,
) -> Iterator[list[ExportedSpeechSegment]]:
    """把一个失败 WAV 临时重切，并将子片段位置换算回原录音。"""

    source_duration_ms = _wav_duration_ms(source_segment.file_path)
    max_speech_duration_ms = math.ceil(
        source_duration_ms / DOUBAO_OVERLAP_FALLBACK_SPLIT_COUNT
    )
    vad_config = VadConfig(
        threshold=vad_threshold,
        min_speech_duration_ms=max(100, max_speech_duration_ms // 2),
        max_speech_duration_ms=max_speech_duration_ms,
    )
    with tempfile.TemporaryDirectory(prefix="podcast-doubao-resplit-") as directory:
        local_segments = detect_and_export_speech_segments(
            source_segment.file_path,
            output_dir=Path(directory),
            config=vad_config,
            silence_padding_ms=silence_padding_ms,
            audio_format=WAV_SEGMENT_AUDIO_FORMAT,
        )
        yield [
            mapped_segment
            for item in local_segments
            if (
                mapped_segment := _map_resplit_segment_to_source(
                    item,
                    source_segment=source_segment,
                    silence_padding_ms=silence_padding_ms,
                )
            )
            is not None
        ]


def _map_resplit_segment_to_source(
    local_segment: ExportedSpeechSegment,
    *,
    source_segment: ExportedSpeechSegment,
    silence_padding_ms: int,
) -> ExportedSpeechSegment | None:
    """去掉源 WAV 自带的静音偏移，恢复子片段在原录音中的位置。"""

    padding_samples = round(silence_padding_ms * VAD_SAMPLE_RATE / 1_000)
    relative_start = max(0, local_segment.segment.start_sample - padding_samples)
    relative_end = max(0, local_segment.segment.end_sample - padding_samples)
    start_sample = min(
        source_segment.segment.end_sample,
        source_segment.segment.start_sample + relative_start,
    )
    end_sample = min(
        source_segment.segment.end_sample,
        source_segment.segment.start_sample + relative_end,
    )
    if end_sample <= start_sample:
        return None
    return ExportedSpeechSegment(
        index=source_segment.index,
        segment=SpeechSegment(
            start_sample=start_sample,
            end_sample=end_sample,
        ),
        file_path=local_segment.file_path,
    )


def _merge_transcription_outputs(
    outputs: Sequence[TranscriptionOutput],
) -> TranscriptionOutput:
    """按播放顺序合并重切子片段的文字和时间信息。"""

    return TranscriptionOutput(
        text="\n".join(output.text for output in outputs),
        character_timestamps=tuple(
            timestamp for output in outputs for timestamp in output.character_timestamps
        ),
        sentences=tuple(
            sentence for output in outputs for sentence in output.sentences
        ),
        diagnostics=outputs[-1].diagnostics,
    )


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as source:
        return round(source.getnframes() * 1_000 / source.getframerate())


@contextmanager
def _temporary_trailing_silence(
    source_path: Path,
    *,
    duration_ms: int,
) -> Iterator[Path]:
    """创建一个只在豆包重试期间使用的临时尾部静音 WAV。"""

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix="podcast-doubao-retry-",
        suffix=".wav",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with wave.open(str(source_path), "rb") as source:
            params = source.getparams()
            frames = source.readframes(source.getnframes())
        silence_frame_count = round(params.framerate * duration_ms / 1_000)
        silence = bytes(silence_frame_count * params.sampwidth * params.nchannels)
        with wave.Wave_write(str(temporary_path)) as target:
            target.setnchannels(params.nchannels)
            target.setsampwidth(params.sampwidth)
            target.setframerate(params.framerate)
            target.writeframes(frames + silence)
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_response_summary(
    response_summary: DoubaoResponseSummary,
    path: Path,
) -> None:
    """确认豆包响应包含可用结果，并且会话已经完整终止。"""

    if not response_summary.is_complete:
        raise DoubaoMissingFinalResponseError(DOUBAO_RESPONSE_ERROR.format(path=path))


def _attempt_quality_key(attempt: _AttemptResult) -> tuple[float, int, float]:
    """生成识别结果的比较标准，供程序选择质量更好的尝试。"""

    return (
        # 有文字对应的人声比例越高越好。
        attempt.assessment.speech_coverage_ratio,
        # 没有文字对应的人声持续时间越短越好，因此取负数参与比较。
        -attempt.assessment.longest_uncovered_speech_ms,
        # 前两项相同时，平均置信度越高越好。
        attempt.mean_confidence,
    )


def _build_attempt_diagnostics(
    attempt: _AttemptResult,
) -> TranscriptionAttemptDiagnostics:
    """把一次尝试整理成可以写入转写文件的检查记录。"""

    return TranscriptionAttemptDiagnostics(
        attempt=attempt.attempt,
        assessment=attempt.assessment,
        raw_responses=(
            attempt.response_summary.raw_responses
            if attempt.assessment.is_anomalous
            else ()
        ),
        error=attempt.error,
    )
