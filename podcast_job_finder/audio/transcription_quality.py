from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.firered_alignment import CharacterAlignment
from podcast_job_finder.audio.normalized_audio import normalize_audio_file
from podcast_job_finder.audio.transcription_diagnostics import (
    MAX_UNCOVERED_SPEECH_MS,
    MIN_SPEECH_COVERAGE_RATIO,
    SpeechCoverageStatus,
    TruncationAssessment,
)
from podcast_job_finder.audio.vad import (
    DEFAULT_VAD_THRESHOLD,
    VAD_FRAME_DURATION_MS,
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    SpeechSegment,
    classify_speech_frames,
)


def assess_transcription_coverage(  # pylint: disable=too-many-arguments
    audio_path: Path,
    alignments: Sequence[CharacterAlignment],
    *,
    has_transcript: bool,
    speech_segment: SpeechSegment | None = None,
    silence_padding_ms: int = 0,
    vad_threshold: float = DEFAULT_VAD_THRESHOLD,
) -> TruncationAssessment:
    speech_frames = _load_speech_frames(
        audio_path,
        speech_segment=speech_segment,
        silence_padding_ms=silence_padding_ms,
        vad_threshold=vad_threshold,
    )
    if not np.any(speech_frames):
        return TruncationAssessment(
            longest_uncovered_speech_ms=0,
            longest_uncovered_speech_start_ms=0,
            longest_uncovered_speech_end_ms=0,
            speech_coverage_ratio=1.0,
            speech_coverage_status=SpeechCoverageStatus.NO_SPEECH,
            has_long_uncovered_speech=False,
        )

    covered_frames = _build_covered_frames(len(speech_frames), alignments)
    longest_start_ms, longest_end_ms = _longest_true_run_range(
        speech_frames & ~covered_frames
    )
    longest_uncovered_ms = longest_end_ms - longest_start_ms
    speech_coverage_ratio = _calculate_speech_coverage(speech_frames, alignments)
    speech_coverage_status = _classify_speech_coverage(
        has_transcript=has_transcript,
        has_alignments=bool(alignments),
        speech_coverage_ratio=speech_coverage_ratio,
    )
    return TruncationAssessment(
        longest_uncovered_speech_ms=longest_uncovered_ms,
        longest_uncovered_speech_start_ms=longest_start_ms,
        longest_uncovered_speech_end_ms=longest_end_ms,
        speech_coverage_ratio=speech_coverage_ratio,
        speech_coverage_status=speech_coverage_status,
        has_long_uncovered_speech=longest_uncovered_ms > MAX_UNCOVERED_SPEECH_MS,
    )


def _classify_speech_coverage(
    *,
    has_transcript: bool,
    has_alignments: bool,
    speech_coverage_ratio: float,
) -> SpeechCoverageStatus:
    if not has_transcript:
        return SpeechCoverageStatus.NO_TRANSCRIPT
    if not has_alignments:
        return SpeechCoverageStatus.ALIGNMENT_FAILED
    if speech_coverage_ratio < MIN_SPEECH_COVERAGE_RATIO:
        return SpeechCoverageStatus.LOW_COVERAGE
    return SpeechCoverageStatus.SUFFICIENT_COVERAGE


def _load_speech_frames(
    audio_path: Path,
    *,
    speech_segment: SpeechSegment | None,
    silence_padding_ms: int,
    vad_threshold: float,
) -> NDArray[np.bool_]:
    """
    定位音频切片内的人声时间段，用于校验语音转写是否出现漏识别。
    前置流程已对完整原始音频完成VAD检测；当前仅处理截取后的音频分片，
    片段里的时间从片段开头开始计算，同时兼容片段前后的静音部分。

    检查点只持久化分片文件与起止时间，不会缓存人声检测结果。
    任务从检查点恢复、跳过全量音频VAD直接加载分片处理时，
    重新读取片段时，人声时间也要从片段开头重新计算。
    两种场景下输出均为相对当前分片本地坐标系的语音点位。

    - 单次传入单个分片可直接调用本方法，内部默认时间原点为 0；
    - speech_segment 批量处理多个分片，各个分片的时间原点并不统一。
    """
    if speech_segment is not None and speech_segment.speech_frames is not None:
        return _build_padded_speech_frames(
            speech_segment,
            silence_padding_ms=silence_padding_ms,
        )

    with normalize_audio_file(audio_path, sample_rate=VAD_SAMPLE_RATE) as audio:
        return classify_speech_frames(
            audio,
            vad_threshold,
            report_progress=False,
        )


def _build_padded_speech_frames(
    segment: SpeechSegment,
    *,
    silence_padding_ms: int,
) -> NDArray[np.bool_]:
    padding_samples = round(silence_padding_ms * VAD_SAMPLE_RATE / 1_000)
    segment_sample_count = segment.end_sample - segment.start_sample
    frame_count = (segment_sample_count + 2 * padding_samples) // VAD_FRAME_SAMPLES
    speech_frames = np.zeros(frame_count, dtype=np.bool_)
    source_frames = segment.speech_frames
    if frame_count == 0 or source_frames is None:
        return speech_frames

    for frame_index in range(frame_count):
        local_start = frame_index * VAD_FRAME_SAMPLES - padding_samples
        local_end = local_start + VAD_FRAME_SAMPLES
        source_start = max(segment.start_sample, segment.start_sample + local_start)
        source_end = min(segment.end_sample, segment.start_sample + local_end)
        if source_end <= source_start:
            continue
        first_frame = max(0, source_start // VAD_FRAME_SAMPLES)
        last_frame = min(
            len(source_frames),
            (source_end + VAD_FRAME_SAMPLES - 1) // VAD_FRAME_SAMPLES,
        )
        speech_frames[frame_index] = bool(np.any(source_frames[first_frame:last_frame]))
    return speech_frames


def _build_covered_frames(
    frame_count: int,
    alignments: Sequence[CharacterAlignment],
) -> NDArray[np.bool_]:
    """标出音频中已经有文字对应的时间小段。

    音频会被分成许多等长的小段。这个函数根据每段文字的开始和结束时间，
    找出哪些小段已经有文字。返回结果中的 ``True`` 表示该小段已有文字，
    ``False`` 表示该小段还没有文字。后续会用它找出“有人说话但没有转成文字”
    的部分。

    人声检测按 16 毫秒一段，文字时间按 40 毫秒一段；两者都从片段开头计算，
    并包含片段前面的静音。当前检查适合发现较长的缺失文字，单个字很短时可能发现不了。
    只有连续超过 2 秒的人声没有对应文字时才会重试；单个字通常只有数百毫秒，
    因此单独漏掉一个字不会触发重试。
    这套实现适合检查片段开头、结尾或连续数秒没有对应文字的情况。

    Args:
        frame_count: 整段音频包含的小段数量。
        alignments: 每段文字及其在音频中的开始、结束时间。

    Returns:
        与音频小段一一对应的标记；已有文字的小段标记为 ``True``。
    """
    # 先把所有音频小段都设为“尚无文字”，之后再逐段更新。
    covered = np.zeros(frame_count, dtype=np.bool_)

    # 依次处理每段文字，找出它对应的音频范围。
    for alignment in alignments:
        # 把文字的开始时间换算成第几个音频小段，并确保位置不会小于 0。
        start_frame = max(0, int(alignment.start_ms // VAD_FRAME_DURATION_MS))

        # 把结束时间也换算成小段位置；只要碰到某个小段，就标记为已有文字。
        # 同时把位置限制在音频总长度内，避免标记到音频范围之外。
        end_frame = min(
            frame_count,
            int(np.ceil(alignment.end_ms / VAD_FRAME_DURATION_MS)),
        )

        # 将这段文字对应的所有音频小段标记为“已有文字”。
        covered[start_frame:end_frame] = True

    # 返回完整标记，供后续查找“有人说话但识别结果里没有文字”的音频片段。
    return covered


def _calculate_speech_coverage(
    speech_frames: NDArray[np.bool_],
    alignments: Sequence[CharacterAlignment],
) -> float:
    """计算“有人说话的部分”有多少已经有对应文字。

    ``speech_frames`` 记录每个时间小段里是否有人说话，
    ``alignments`` 记录每个字在音频中的开始和结束位置。
    函数会找到文字对应的音频范围，再统计这段范围里有多少人声，
    最后用“已有文字的人声 / 全部人声”得到一个 0 到 1 之间的比例。

    例如，音频里一共有 10 秒人声，其中 8 秒有对应文字，返回值就是 0.8。
    没有任何文字时间信息时，表示没有一段人声能和文字对应，返回值为 0。
    """
    if not alignments:
        # 没有文字时间位置可供比较，直接视为没有人声有对应文字。
        return 0.0

    # 找到第一段文字开始的时间，换算成语音帧的位置。
    # 这样可以跳过文字出现前的静音或未转写区域。
    start_frame = max(
        0,
        int(min(item.start_ms for item in alignments) // VAD_FRAME_DURATION_MS),
    )

    # 找到最后一段文字结束的时间，作为文字对应范围的末尾。
    # 文字结束时间落在某个小段中间时，也把这个小段算作已有文字。
    end_frame = min(
        len(speech_frames),
        int(np.ceil(max(item.end_ms for item in alignments) / VAD_FRAME_DURATION_MS)),
    )

    # 统计整段音频里实际有人说话的时间小段数量。
    speech_count = int(np.count_nonzero(speech_frames))

    # 只统计已经有文字对应的人声小段，得到已经识别出来的部分。
    covered_speech_count = int(np.count_nonzero(speech_frames[start_frame:end_frame]))

    # 返回对应比例：1 表示全部人声都有文字，0 表示没有人声有文字。
    return covered_speech_count / speech_count


def _longest_true_run_range(values: NDArray[np.bool_]) -> tuple[int, int]:
    longest_start = 0
    longest_end = 0
    current_start: int | None = None
    for index, value in enumerate(values):
        if value and current_start is None:
            current_start = index
            continue
        if value or current_start is None:
            continue
        if index - current_start > longest_end - longest_start:
            longest_start = current_start
            longest_end = index
        current_start = None
    if (
        current_start is not None
        and len(values) - current_start > longest_end - longest_start
    ):
        longest_start = current_start
        longest_end = len(values)
    return (
        round(longest_start * VAD_FRAME_DURATION_MS),
        round(longest_end * VAD_FRAME_DURATION_MS),
    )
