from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Final

import numpy as np
from numpy.typing import NDArray

from podcast_job_finder.audio.segmentation._pcm import milliseconds_to_frames
from podcast_job_finder.audio.segmentation._segment_candidates import (
    BoundarySearchConfig,
    SegmentBoundary,
    build_segment_boundaries,
)
from podcast_job_finder.audio.segmentation.normalized_audio import NormalizedAudio


# 重叠长度需要小于这个比例对应的帧数，避免重叠覆盖大部分片段。
MIN_CUT_POSITION_RATIO: Final = 0.25

# 停顿达到 300 毫秒后，继续变长不会增加停顿完整度得分。
FULL_SILENCE_DURATION_MS: Final = 300

# 边界质量优先参考停顿完整度，安静程度用于区分长度接近的停顿。
SILENCE_DURATION_SCORE_WEIGHT: Final = 2 / 3
SILENCE_ENERGY_SCORE_WEIGHT: Final = 1 / 3

# 动态规划损失的固定权重。短片段损失较高，用于替代后置贪心合并。
BOUNDARY_QUALITY_LOSS_WEIGHT: Final = 0.75
SEGMENT_LENGTH_LOSS_WEIGHT: Final = 0.25
SHORT_SEGMENT_LOSS_WEIGHT: Final = 1.0
ADDITIONAL_SEGMENT_LOSS: Final = 0.05


@dataclass(slots=True, frozen=True)
class SegmentPartitionConfig:
    min_speech_frames: int
    max_speech_frames: int
    overlap_frames: int
    frame_samples: int


@dataclass(slots=True)
class _PartitionState:
    costs: list[float]
    segment_counts: list[int]
    previous_indexes: list[int]


def optimize_segment_partition(
    segments: list[tuple[int, int]],
    *,
    speech_frames: NDArray[np.bool_],
    audio: NormalizedAudio,
    config: SegmentPartitionConfig,
) -> list[tuple[int, int]]:
    """用动态规划同时完成长片段切分和相邻短片段合并。"""
    if not segments:
        return []

    boundaries = build_segment_boundaries(
        segments,
        speech_frames=speech_frames,
        audio=audio,
        config=BoundarySearchConfig(
            max_speech_frames=config.max_speech_frames,
            overlap_frames=config.overlap_frames,
            frame_samples=config.frame_samples,
        ),
    )
    boundary_losses = _calculate_boundary_losses(
        boundaries,
        sample_rate=audio.sample_rate,
        config=config,
    )
    selected_indexes = _find_optimal_boundary_indexes(
        boundaries,
        boundary_losses=boundary_losses,
        config=config,
    )
    return _build_segments(boundaries, selected_indexes)


def _calculate_boundary_losses(
    boundaries: list[SegmentBoundary],
    *,
    sample_rate: int,
    config: SegmentPartitionConfig,
) -> list[float]:
    internal_boundaries = [
        boundary for boundary in boundaries if not boundary.is_natural
    ]
    if internal_boundaries:
        min_energy = min(boundary.mean_energy for boundary in internal_boundaries)
        max_energy = max(boundary.mean_energy for boundary in internal_boundaries)
    else:
        min_energy = max_energy = 0.0

    full_silence_frames = milliseconds_to_frames(
        FULL_SILENCE_DURATION_MS,
        sample_rate=sample_rate,
        frame_samples=config.frame_samples,
    )
    return [
        _calculate_boundary_loss(
            boundary,
            min_energy=min_energy,
            max_energy=max_energy,
            full_silence_frames=full_silence_frames,
        )
        for boundary in boundaries
    ]


def _calculate_boundary_loss(
    boundary: SegmentBoundary,
    *,
    min_energy: float,
    max_energy: float,
    full_silence_frames: int,
) -> float:
    if boundary.is_natural:
        return ADDITIONAL_SEGMENT_LOSS

    duration_score = min(
        1.0,
        boundary.silence_duration_frames / full_silence_frames,
    )
    quietness_score = _normalize_quietness(
        boundary.mean_energy,
        min_energy=min_energy,
        max_energy=max_energy,
    )
    quality_score = (
        duration_score * SILENCE_DURATION_SCORE_WEIGHT
        + quietness_score * SILENCE_ENERGY_SCORE_WEIGHT
    )
    return (
        ADDITIONAL_SEGMENT_LOSS + (1.0 - quality_score) * BOUNDARY_QUALITY_LOSS_WEIGHT
    )


def _normalize_quietness(
    energy: float,
    *,
    min_energy: float,
    max_energy: float,
) -> float:
    if max_energy == min_energy:
        return 1.0
    return 1.0 - (energy - min_energy) / (max_energy - min_energy)


def _find_optimal_boundary_indexes(
    boundaries: list[SegmentBoundary],
    *,
    boundary_losses: list[float],
    config: SegmentPartitionConfig,
) -> list[int]:
    state = _PartitionState(
        costs=[inf] * len(boundaries),
        segment_counts=[0] * len(boundaries),
        previous_indexes=[-1] * len(boundaries),
    )
    state.costs[0] = 0.0

    for end_index in range(1, len(boundaries)):
        _update_optimal_predecessor(
            end_index,
            boundaries=boundaries,
            boundary_losses=boundary_losses,
            config=config,
            state=state,
        )

    if state.previous_indexes[-1] < 0:
        raise RuntimeError("动态规划无法生成满足最长时长约束的语音片段。")
    return _backtrack_boundary_indexes(state.previous_indexes)


def _update_optimal_predecessor(
    end_index: int,
    *,
    boundaries: list[SegmentBoundary],
    boundary_losses: list[float],
    config: SegmentPartitionConfig,
    state: _PartitionState,
) -> None:
    best_rank = (inf, len(boundaries), 0)
    end_frame = boundaries[end_index].previous_end_frame
    is_terminal = end_index == len(boundaries) - 1
    for start_index in range(end_index - 1, -1, -1):
        # 最长时长约束只计算本段新增的原始音频；切点重叠会在结果构建时额外加入。
        segment_frames = end_frame - boundaries[start_index].next_start_frame
        if segment_frames > config.max_speech_frames:
            break
        if segment_frames <= 0 or state.costs[start_index] == inf:
            continue

        candidate_cost = state.costs[start_index] + _calculate_segment_loss(
            segment_frames,
            config=config,
        )
        if not is_terminal:
            candidate_cost += boundary_losses[end_index]
        candidate_count = state.segment_counts[start_index] + 1
        candidate_rank = (candidate_cost, candidate_count, -segment_frames)
        if candidate_rank >= best_rank:
            continue
        best_rank = candidate_rank
        state.costs[end_index] = candidate_cost
        state.segment_counts[end_index] = candidate_count
        state.previous_indexes[end_index] = start_index


def _calculate_segment_loss(
    segment_frames: int,
    *,
    config: SegmentPartitionConfig,
) -> float:
    target_frames = (config.min_speech_frames + config.max_speech_frames) / 2
    length_loss = abs(segment_frames - target_frames) / target_frames
    shortfall_ratio = max(
        0.0,
        (config.min_speech_frames - segment_frames) / config.min_speech_frames,
    )
    return (
        length_loss * SEGMENT_LENGTH_LOSS_WEIGHT
        + shortfall_ratio * SHORT_SEGMENT_LOSS_WEIGHT
    )


def _backtrack_boundary_indexes(previous_indexes: list[int]) -> list[int]:
    selected_indexes = [len(previous_indexes) - 1]
    current_index = selected_indexes[0]
    while current_index > 0:
        current_index = previous_indexes[current_index]
        selected_indexes.append(current_index)
    selected_indexes.reverse()
    return selected_indexes


def _build_segments(
    boundaries: list[SegmentBoundary],
    selected_indexes: list[int],
) -> list[tuple[int, int]]:
    segments = []
    for position, end_index in enumerate(selected_indexes[1:]):
        start_index = selected_indexes[position]
        start_frame = boundaries[start_index].output_start_frame
        end_frame = boundaries[end_index].previous_end_frame
        segments.append((start_frame, end_frame))
    return segments
