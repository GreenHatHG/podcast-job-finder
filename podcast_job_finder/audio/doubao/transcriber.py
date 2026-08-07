"""豆包音频转写的主流程。

这个文件负责把一个音频片段从“发送给豆包”处理到“整理成项目可以保存的结果”。
它还会在识别结果疑似漏内容时，检查可疑位置并决定是否完整重试一次。
"""

from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from podcast_job_finder.audio.firered_alignment import (
    CharacterAlignment,
    FireRedTextAlignmentClient,
)
from podcast_job_finder.audio.segment_export import ExportedSpeechSegment
from podcast_job_finder.audio.transcription import (
    AudioTranscriptionError,
    TranscriptionSegmentResult,
    TranscribedSpeechSegment,
    TranscriptionOutput,
    build_transcribed_speech_segment,
)
from podcast_job_finder.audio.transcription_diagnostics import (
    TranscriptionAttemptDiagnostics,
    TranscriptionDiagnostics,
    TruncationProbeDiagnostics,
    TruncationAssessment,
)
from podcast_job_finder.audio.transcription_quality import (
    assess_transcription_coverage,
)

from .client import build_doubao_client
from .config import DOUBAO_MAX_ATTEMPTS, DoubaoTranscriberConfig
from .output import build_doubao_transcription_output
from .request_client import DoubaoRequestClient, SessionResponses
from .response import (
    AsrResponseProtocol,
    DOUBAO_RESPONSE_ERROR,
    DoubaoResponseSummary,
    build_doubao_response_summary,
)
from .truncation_probe import (
    DoubaoTruncationProbeRunner,
    TruncationProbeRequest,
    log_truncation_probe_result,
    probe_requires_full_retry,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _AttemptResult:
    """记录一次识别尝试的完整结果。

    同一个片段最多可能有两次尝试：第一次是正常识别，第二次是发现“识别结果里
    有一段人声没有对应文字”后的完整重试。这里把文字、每个字出现的时间和检查
    结果放在一起，方便最后选择更可靠的一次。
    """

    # 第几次识别，从 1 开始。
    attempt: int
    # 从豆包返回的多条消息中整理出的文字和检查结果。
    response_summary: DoubaoResponseSummary
    # 每个字在音频中的开始、结束时间，以及程序认为它识别得有多可靠。
    alignments: tuple[CharacterAlignment, ...]
    # 用来判断是否有一段人声没有对应文字。
    assessment: TruncationAssessment

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

    # 通常只有第一次尝试；确认有一段人声没有对应文字后会追加完整重试。
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


class DoubaoAudioTranscriber:
    """负责请求豆包、检查结果，并整理出带时间信息的文字。"""

    def __init__(self, config: DoubaoTranscriberConfig) -> None:
        """创建转写器及其所需的辅助组件。"""

        self._config = config

        # FireRed 用来找出每个字在音频里的开始和结束时间。
        self._aligner = FireRedTextAlignmentClient(config.alignment_config)

        # 豆包客户端负责发请求；这里同时拿到响应类型，后面用它判断最终结果。
        (
            asr_config,
            transcribe_stream,
            self._response_type,
        ) = build_doubao_client()
        self._request_client = DoubaoRequestClient(
            asr_config=asr_config,
            transcribe_stream=transcribe_stream,
            max_in_flight_requests=config.max_in_flight_requests,
            request_interval_seconds=config.request_interval_seconds,
        )

        # 初次结果里可能有一段人声没有对应文字时，单独识别那段音频进行确认。
        self._probe_runner = DoubaoTruncationProbeRunner(
            collect_responses=self._request_client.collect_probe_responses,
            response_types=self._response_type,
        )

    @property
    def batch_size(self) -> int:
        """告诉上层一次最多可以同时等待多少个豆包请求。"""

        return self._config.max_in_flight_requests

    def transcribe(
        self,
        segment: ExportedSpeechSegment,
        *,
        previous_segment: TranscribedSpeechSegment | None,
    ) -> TranscriptionOutput:
        """按顺序处理一个片段，并返回可保存的转写结果。

        处理顺序是：初次识别、检查结果里是否有一段人声没有对应文字、必要时
        完整重试，最后去除和前一个片段重复的内容并生成时间信息。
        """

        # 先完成一次正常识别，后面的判断都基于这次结果进行。
        initial_responses = self._request_client.collect_responses(segment.file_path)
        attempts = self._run_segment_attempts(segment, initial_responses)

        # 选择更完整可靠的结果，并换算成它在原始音频中的实际时间。
        return self._build_transcription_output(
            segment,
            attempts,
            previous_segment=previous_segment,
        )

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

        # 记住最早发生的错误；已提交的请求处理完后，再把错误交给上层。
        first_error: Exception | None = None
        for segment in segments:
            # 提交请求后立即继续提交下一个片段，让多个请求可以同时进行。
            future = self._request_client.submit_segment(segment)
            pending_requests.append(_PendingRequest(segment, future))

            # 队列未达到上限时继续提交，达到上限后才取出最早的请求。
            if len(pending_requests) < self._config.max_in_flight_requests:
                continue

            item, current_previous, error = self._consume_pending_request(
                pending_requests.popleft(),
                previous_segment=current_previous,
            )
            if item is not None:
                # 成功结果马上交给上层，上层可以立刻写入片段检查点。
                yield (item,)
            if error is not None:
                # 停止提交新请求，已经提交的请求仍然会继续处理。
                first_error = error
                break

        # 处理剩余的已提交请求，尽量保存其中已经成功的结果。
        while pending_requests:
            item, current_previous, error = self._consume_pending_request(
                pending_requests.popleft(),
                previous_segment=current_previous,
            )
            if item is not None:
                yield (item,)
            if error is not None and first_error is None:
                first_error = error

        # 所有已提交请求都处理完后，再把最早的错误报告给上层。
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

        try:
            # 等待后台请求完成，然后取得豆包这次返回的全部消息。
            responses = pending_request.future.result()
        except Exception as error:  # pylint: disable=broad-exception-caught
            # 失败片段仍然保留时间范围，后面的片段可以继续去掉重复音频。
            return (
                None,
                _build_failed_segment_context(pending_request.segment),
                error,
            )
        try:
            # 整理返回消息，检查是否有一段人声没有对应文字，并在需要时重试。
            item = self._build_transcription_item(
                pending_request.segment,
                responses,
                previous_segment=previous_segment,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            # 结果整理阶段也可能失败，按单个片段失败处理。
            return (
                None,
                _build_failed_segment_context(pending_request.segment),
                error,
            )
        return (
            item,
            # 下一片段需要使用当前片段的完整结果作为参考。
            build_transcribed_speech_segment(pending_request.segment, item.output),
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

        attempts = self._run_segment_attempts(segment, responses)
        return TranscriptionSegmentResult(
            segment=segment,
            output=self._build_transcription_output(
                segment,
                attempts,
                previous_segment=previous_segment,
            ),
            previous_segment=previous_segment,
        )

    def _run_segment_attempts(
        self,
        segment: ExportedSpeechSegment,
        initial_responses: SessionResponses,
    ) -> _SegmentAttempts:
        """完成一个片段从初次识别到可选重试的完整流程。

        初次识别结果里的每段人声都有对应文字时直接结束。如果发现有一段人声
        没有对应文字，就单独识别那段音频；确认那里确实有人说话，或单独识别
        过程发生错误时，再完整重试当前片段。
        """

        # 把第一次返回的消息整理成文字、每个字出现的时间和检查结果。
        initial_attempt = self._build_attempt_result(
            segment,
            initial_responses,
            attempt=1,
        )
        attempts = _SegmentAttempts([initial_attempt])

        # 结果正常时不做额外请求，避免增加服务压力和处理时间。
        if not initial_attempt.assessment.requires_truncation_probe:
            return attempts

        # 单独识别没有对应文字的那段音频，确认那里是否真的有人说话。
        attempts.probe = self._probe_runner.run(
            TruncationProbeRequest(
                audio_path=segment.file_path,
                assessment=initial_attempt.assessment,
            )
        )
        log_truncation_probe_result(
            segment_index=segment.index,
            assessment=initial_attempt.assessment,
            probe=attempts.probe,
        )
        if probe_requires_full_retry(attempts.probe):
            # 探测确认需要重试时，重新识别完整片段，而不是只识别局部音频。
            attempts.attempts.append(
                self._run_attempt(segment, attempt=DOUBAO_MAX_ATTEMPTS)
            )
        return attempts

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
            silence_padding_ms=self._config.silence_padding_ms,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        """关闭请求调度器和负责查找文字时间的进程。"""

        self._request_client.close()
        self._aligner.close()

    def _run_attempt(
        self,
        segment: ExportedSpeechSegment,
        *,
        attempt: int,
    ) -> _AttemptResult:
        """重新请求整个片段，并整理这次请求的结果。"""

        responses = self._request_client.collect_responses(segment.file_path)
        return self._build_attempt_result(segment, responses, attempt=attempt)

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
        )
        # 豆包没有返回“识别完成”的消息时，这次结果无法使用。
        _validate_response_summary(response_summary, segment.file_path)
        if not response_summary.text:
            logger.info(
                "豆包片段未识别到文字，保存空结果：index=%d path=%s",
                segment.index,
                segment.file_path,
            )
        alignments = (
            # 没有文字时无需查找每个字的时间，直接检查音频里是否有人说话。
            self._aligner.align(segment.file_path, response_summary.text)
            if response_summary.text
            else ()
        )
        # 比较“有人说话的时间”和“文字出现的时间”，找出是否有一段人声没有文字。
        assessment = assess_transcription_coverage(
            segment.file_path,
            alignments,
            has_transcript=bool(response_summary.text),
            speech_segment=segment.segment,
            silence_padding_ms=self._config.silence_padding_ms,
            vad_threshold=self._config.vad_threshold,
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
    """为失败片段创建一个只有时间边界的占位信息。

    后续片段仍然需要知道这个片段在原音频中的结束位置，才能去除相邻片段
    之间重复的音频内容。
    """

    return TranscribedSpeechSegment(
        index=segment.index,
        start_ms=segment.segment.start_ms,
        end_ms=segment.segment.end_ms,
        text="",
    )


def _validate_response_summary(
    response_summary: DoubaoResponseSummary,
    path: Path,
) -> None:
    """确认豆包响应中包含最终识别结果。"""

    if not response_summary.has_final_response:
        raise AudioTranscriptionError(DOUBAO_RESPONSE_ERROR.format(path=path))


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
    )
