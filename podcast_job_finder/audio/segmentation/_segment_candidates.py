from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.segmentation._segment_candidate_search import (
    CutCandidate,
    build_fallback_candidates,
    build_silence_candidates,
    deduplicate_candidates,
)
from podcast_job_finder.audio.segmentation.normalized_audio import NormalizedAudio


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
class SegmentPartitionConfig:
    min_speech_frames: int
    max_speech_frames: int
    overlap_frames: int
    frame_samples: int


def build_segment_boundaries(
    segments: list[tuple[int, int]],
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    config: SegmentPartitionConfig,
) -> list[SegmentBoundary]:
    """整理出所有可供后续分段使用的边界。

    函数先按照第一段语音起点到最后一段语音终点的完整范围，统一准备控制
    最长时长所需的强制切点。然后逐个处理自然语音段，加入段内停顿和段间
    自然停顿。这里仅准备可选边界，最终使用哪些边界由后续动态规划统一决定。
    """
    # 起点也作为一个边界，方便后续从第一个位置开始组合语音片段。
    boundaries = [_terminal_boundary(segments[0][0])]

    # 强制切点必须在进入循环前按照完整语音范围统一生成。如果在循环中为每个
    # 自然语音段重新生成，各段都会从自己的起点重新平均切分，前一段留下的长度
    # 无法影响后一段，节目开头或结尾就可能剩下只有几百毫秒的独立短片段。
    # 统一生成后，所有自然语音段使用同一组位置参考，动态规划可以跨越自然停顿
    # 重新安排片段长度，同时仍可优先选择后面加入的真实停顿位置。
    fallback_candidates = build_fallback_candidates(
        audio,
        start_frame=segments[0][0],
        end_frame=segments[-1][1],
        max_speech_frames=config.max_speech_frames,
        frame_samples=config.frame_samples,
    )

    for segment_index, (start_frame, end_frame) in enumerate(segments):
        # 一段连续讲话可能仍然很长，这里会在它的内部完成以下操作：
        # 1. 寻找适合自然切开的短暂停顿；
        # 2. 按照目标片段长度准备备用切点，避免单段音频过长；
        # 3. 合并落在同一位置的重复候选，保留停顿更长或声音更安静的一个；
        # 4. 把候选整理成统一的边界，并为缺少停顿的备用切点保留少量重叠音频。
        # 找到的所有内部边界会加入 boundaries，供后续流程统一选择。
        boundaries.extend(
            _build_internal_boundaries(
                (start_frame, end_frame),
                speech_frames=speech_frames,
                audio=audio,
                config=config,
                fallback_candidates=fallback_candidates,
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
    segment: tuple[int, int],
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    config: SegmentPartitionConfig,
    fallback_candidates: list[CutCandidate],
) -> list[SegmentBoundary]:
    """为一段连续语音寻找可用的内部切分位置。

    函数先找出当前自然语音段里的真实短暂停顿，再从整段语音统一生成的
    fallback_candidates 中取出落在当前范围内的强制切点。返回的每一项都
    描述上一片段在哪里结束，以及下一片段应该从哪里开始。
    """
    start_frame, end_frame = segment

    # 优先寻找讲话中的短暂停顿。这些位置更接近自然停顿，切开后听感更顺畅。
    candidates = build_silence_candidates(
        speech_frames,
        audio,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_samples=config.frame_samples,
    )

    # fallback_candidates 同时包含整段语音各处的强制切点。当前循环只负责
    # 包装这个自然语音段内部的切点；落在其他语音段的切点会在对应循环中处理，
    # 落在两段语音之间安静部分的切点则不会使用。这样既保留统一的长度安排，
    # 又不会为了让片段长度接近而在一段长时间安静的地方额外制造强制边界。
    candidates.extend(
        candidate
        for candidate in fallback_candidates
        if start_frame < candidate.cut_frame < end_frame
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
