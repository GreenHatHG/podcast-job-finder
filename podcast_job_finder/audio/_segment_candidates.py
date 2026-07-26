from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio._segment_candidate_search import (
    CutCandidate,
    build_fallback_candidates,
    build_silence_candidates,
    deduplicate_candidates,
)
from podcast_job_finder.audio.normalized_audio import NormalizedAudio


@dataclass(slots=True, frozen=True)
class SegmentBoundary:
    """描述相邻输出片段之间可以选择的一条边界。"""

    previous_end_frame: int
    next_start_frame: int
    output_start_frame: int
    silence_duration_frames: int
    mean_energy: float
    is_natural: bool


@dataclass(slots=True, frozen=True)
class BoundarySearchConfig:
    max_speech_frames: int
    overlap_frames: int
    frame_samples: int


def build_segment_boundaries(
    segments: list[tuple[int, int]],
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    config: BoundarySearchConfig,
) -> list[SegmentBoundary]:
    """整理出所有可供后续分段使用的边界。

    可以把一段较长的录音想成一篇没有分段的文章。这个函数会先找出所有
    适合“换段”的位置，例如两段话之间原本就有的停顿，以及一段连续讲话中
    相对适合切开的地方。它只负责准备候选位置，最终选用哪些位置由后续逻辑决定。
    """
    # 起点也作为一个边界，方便后续从第一个位置开始组合语音片段。
    boundaries = [_terminal_boundary(segments[0][0])]

    for segment_index, (start_frame, end_frame) in enumerate(segments):
        # 一段连续讲话可能仍然很长，这里会在它的内部完成以下操作：
        # 1. 寻找适合自然切开的短暂停顿；
        # 2. 按照目标片段长度准备备用切点，避免单段音频过长；
        # 3. 合并落在同一位置的重复候选，保留停顿更长或声音更安静的一个；
        # 4. 把候选整理成统一的边界，并为缺少停顿的备用切点保留少量重叠音频。
        # 找到的所有内部边界会加入 boundaries，供后续流程统一选择。
        boundaries.extend(
            _build_internal_boundaries(
                start_frame,
                end_frame,
                speech_frames=speech_frames,
                audio=audio,
                config=config,
            )
        )

        # 确认当前段后面还有下一段
        # 相邻语音段之间本来就有停顿，这类位置通常很适合直接作为分段边界。
        # 前面的步骤拆分过一次，切分的规则就是按照有没有自然停顿来切分的，
        # 这样自然每一段都是自然停顿
        if segment_index + 1 < len(segments):
            next_start = segments[segment_index + 1][0]
            boundaries.append(_natural_boundary(end_frame, next_start))

    # 加入终点后，候选边界就完整覆盖了从第一段语音到最后一段语音的范围。
    boundaries.append(_terminal_boundary(segments[-1][1]))
    return boundaries


def _terminal_boundary(frame: int) -> SegmentBoundary:
    return SegmentBoundary(
        previous_end_frame=frame,
        next_start_frame=frame,
        output_start_frame=frame,
        silence_duration_frames=0,
        mean_energy=0.0,
        is_natural=True,
    )


def _natural_boundary(
    previous_end_frame: int,
    next_start_frame: int,
) -> SegmentBoundary:
    return SegmentBoundary(
        previous_end_frame=previous_end_frame,
        next_start_frame=next_start_frame,
        output_start_frame=next_start_frame,
        silence_duration_frames=next_start_frame - previous_end_frame,
        mean_energy=0.0,
        is_natural=True,
    )


def _build_internal_boundaries(
    start_frame: int,
    end_frame: int,
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    config: BoundarySearchConfig,
) -> list[SegmentBoundary]:
    """为一段连续语音寻找可用的内部切分位置。

    有些连续讲话很长，中间没有现成的语音段边界。这个函数会先寻找适合
    切开的短暂停顿，再补充一些备用切点，避免最终得到的音频片段过长。
    返回的每一项都描述一个切分位置，以及下一段音频应该从哪里开始。
    """
    # 优先寻找讲话中的短暂停顿。这些位置更接近自然停顿，切开后听感更顺畅。
    candidates = build_silence_candidates(
        speech_frames,
        audio,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_samples=config.frame_samples,
    )

    # 连续讲话可能完全没有可用停顿，因此再加入按照片段长度准备的备用切点。
    # 后续逻辑可以在找不到理想停顿时使用它们，防止单个输出片段过长。
    candidates.extend(
        build_fallback_candidates(
            audio,
            start_frame=start_frame,
            end_frame=end_frame,
            max_speech_frames=config.max_speech_frames,
            frame_samples=config.frame_samples,
        )
    )

    # 两种来源可能给出同一个切点。先去重，再把每个候选位置整理成后续流程
    # 统一使用的边界对象。
    return [
        SegmentBoundary(
            # 这里在连续语音内部直接切开，上一段结束和下一段开始使用同一切点。
            previous_end_frame=candidate.cut_frame,
            next_start_frame=candidate.cut_frame,
            # 下一段的实际输出起点可能稍微提前，以保留必要的重叠音频。
            output_start_frame=_calculate_output_start_frame(
                candidate,
                segment_start_frame=start_frame,
                overlap_frames=config.overlap_frames,
            ),
            # 保存停顿长度和声音大小，供后续选择更合适的边界。
            silence_duration_frames=candidate.silence_duration_frames,
            mean_energy=candidate.mean_energy,
            # 这些边界来自一段连续语音的内部，因此不属于原本就存在的段间边界。
            is_natural=False,
        )
        for candidate in deduplicate_candidates(candidates)
    ]


def _calculate_output_start_frame(
    candidate: CutCandidate,
    *,
    segment_start_frame: int,
    overlap_frames: int,
) -> int:
    # 停顿切点已经保护了前后语音，直接从切点开始可避免带入残缺词尾。
    if candidate.silence_duration_frames > 0:
        return candidate.cut_frame

    # 连续语音中的备用切点仍保留重叠，降低从词语中间切开的影响。
    return max(segment_start_frame, candidate.cut_frame - overlap_frames)
